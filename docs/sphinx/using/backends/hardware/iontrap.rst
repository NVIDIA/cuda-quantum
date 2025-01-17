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

   To see a complete example for using IonQ's backends, take a look at our :doc:`Python examples <../../examples/examples>`.


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

        To see a complete example for using IonQ's backends, take a look at our :doc:`C++ examples <../../examples/examples>`.
  

Quantinuum
+++++++++++

.. _quantinuum-backend:

Setting Credentials
```````````````````

Programmers of CUDA-Q may access the Quantinuum API from either
C++ or Python. Quantinuum requires a credential configuration file. 
The configuration file can be generated as follows, replacing
the ``email`` and ``credentials`` in the first line with your Quantinuum
account details.

.. code:: bash

    # You may need to run: `apt-get update && apt-get install curl jq`
    curl -X POST -H "Content Type: application/json" \
        -d '{ "email":"<your_alias>@email.com","password":"<your_password>" }' \
        https://qapi.quantinuum.com/v1/login > $HOME/credentials.json
    id_token=`cat $HOME/credentials.json | jq -r '."id-token"'`
    refresh_token=`cat $HOME/credentials.json | jq -r '."refresh-token"'`
    echo "key: $id_token" >> $HOME/.quantinuum_config
    echo "refresh: $refresh_token" >> $HOME/.quantinuum_config

The path to the configuration can be specified as an environment variable:

.. code:: bash

    export CUDAQ_QUANTINUUM_CREDENTIALS=$HOME/.quantinuum_config


Submitting
`````````````````````````
.. tab:: Python

       
        The backend to which quantum kernels are submitted 
        can be controlled with the ``cudaq::set_target()`` function.

        .. code:: python

            cudaq.set_target('quantinuum')

        By default, quantum kernel code will be submitted to the Quantinuum syntax checker.
        Submission to the syntax checker merely validates the program; the kernels are not executed.

        To execute your kernels, specify which machine to submit quantum kernels to
        by setting the :code:`machine` parameter of the target.

        .. code:: python

            cudaq.set_target('quantinuum', machine='H1-2')

        where ``H1-2`` is an example of a physical QPU. Hardware specific
        emulators may be accessed by appending an ``E`` to the end (e.g, ``H1-2E``). For 
        access to the syntax checker for the provided machine, you may append an ``SC`` 
        to the end (e.g, ``H1-1SC``).

        For a comprehensive list of available machines, login to your `Quantinuum user account <https://um.qapi.quantinuum.com/user>`__ 
        and navigate to the "Account" tab, where you should find a table titled "Machines".

        To emulate the Quantinuum machine locally, without submitting through the cloud,
        you can also set the ``emulate`` flag to ``True``. This will emit any target 
        specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: python

            cudaq.set_target('quantinuum', emulate=True)

        The number of shots for a kernel execution can be set through
        the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
        the ``shots_count`` is set to 1000.

        .. code:: python 

            cudaq.sample(kernel, shots_count=10000)

        To see a complete example for using Quantinuum's backends, take a look at our :doc:`Python examples <../../examples/examples>`.


.. tab:: C++

        To target quantum kernel code for execution in the Quantinuum backends,
        pass the flag ``--target quantinuum`` to the ``nvq++`` compiler. CUDA-Q will 
        authenticate via the Quantinuum REST API using the credential in your configuration file.
        By default, quantum kernel code will be submitted to the Quantinuum syntax checker.
        Submission to the syntax checker merely validates the program; the kernels are not executed.

        .. code:: bash

            nvq++ --target quantinuum src.cpp ...

        To execute your kernels, pass the ``--quantinuum-machine`` flag to the ``nvq++`` compiler
        to specify which machine to submit quantum kernels to:

        .. code:: bash

            nvq++ --target quantinuum --quantinuum-machine H1-2 src.cpp ...

        where ``H1-2`` is an example of a physical QPU. Hardware specific
        emulators may be accessed by appending an ``E`` to the end (e.g, ``H1-2E``). For 
        access to the syntax checker for the provided machine, you may append an ``SC`` 
        to the end (e.g, ``H1-1SC``).

        For a comprehensive list of available machines, login to your `Quantinuum user account <https://um.qapi.quantinuum.com/user>`__ 
        and navigate to the "Account" tab, where you should find a table titled "Machines".

        To emulate the Quantinuum machine locally, without submitting through the cloud,
        you can also pass the ``--emulate`` flag to ``nvq++``. This will emit any target 
        specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: bash

            nvq++ --emulate --target quantinuum src.cpp

        To see a complete example for using Quantinuum's backends, take a look at our :doc:`C++ examples <../../examples/examples>`.

