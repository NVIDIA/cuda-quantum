CUDA Quantum Hardware Backends
*********************************

CUDA Quantum supports submission to a set of hardware providers. 
To submit to a hardware backend, you need an account with the respective provider.

IonQ
==================================

Setting Credentials
```````````````````

Programmers of CUDA Quantum may access the `IonQ Quantum Cloud
<https://cloud.ionq.com/>`_ from either C++ or Python. Simply generate
an API key from your `IonQ account <https://cloud.ionq.com/>`_ and export
it as an environment variable:

.. code:: bash

  export IONQ_API_KEY="ionq_generated_api_key"


C++
```

For developers in C++, you can indicate to ``nvq++`` that your quantum
kernels will be executed in the IonQ Cloud via the ``--target`` flag.

.. code:: bash

    nvq++ --target ionq src.cpp ...

This will take the API key and handle all authentication with, and submission to,
the IonQ QPU.

Note: A "target" in :code:`cudaq` refers to a quantum compute provider, such as :code:`ionq`.
However, IonQ's documentation uses the term "target" to refer to specific QPU's themselves.

Specifying the QPU
'''''''''''''''''''

At this time, programmer control over the IonQ QPU is under construction in ``nvq++``.


Example
.......

.. literalinclude:: ../examples/cpp/ionq
    :language: cpp


Python
```````

For python developers, the target may be controlled with the ``cudaq::set_target()``
function. This is functionally equivalent to the ``nvq++`` target,
and will handle the submission of all quantum kernels to IonQ.

.. code:: python

    cudaq.set_target('ionq')

To emulate the IonQ machine locally, without submitting through the cloud,
you can also set the ``emulate`` flag to ``True``. This will emit any target 
specific compiler warnings and diagnostics, before running a noise free emulation.

.. code:: python

    cudaq.set_target('ionq', emulate=True)

To select the number of shots for the kernel execution, this may be done through
the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
the ``shots_count`` is set to 1000.

.. code:: python 

    cudaq.sample(kernel, shots_count=10000)

Note: A "target" in :code:`cudaq` refers to a quantum compute provider, such as :code:`ionq`.
However, IonQ's documentation uses the term "target" to refer to specific QPU's themselves.

Specifying the QPU
'''''''''''''''''''

By default, the IonQ target will use the :code:`simulator` QPU.
To specify which IonQ QPU to use, set the :code:`qpu` parameter.
A list of available QPUs can be found `in the API documentation
<https://docs.ionq.com/#tag/jobs>`_.

.. code:: python

    cudaq.set_target("ionq", qpu="qpu.aria-1")

Example
........

.. literalinclude:: ../examples/python/ionq
   :language: python


Quantinuum
==================================

Setting Credentials
```````````````````

Programmers of CUDA Quantum may access the Quantinuum API from either
C++ or Python. Quantinuum requires a credential configuration file
in your ``$HOME`` directory. This may be generated as follows, replacing
the ``email`` and ``credentials`` in the first line with your Quantinuum
account details.

.. code:: bash

    # You may need to run: `apt-get update && apt-get install curl jq`
    curl -X POST -H "Content Type: application/json" -d '{ "email":"<your_alias>@email.com","password":"<your_password>" }' https://qapi.quantinuum.com/v1/login > $HOME/credentials.json
    id_token=`cat $HOME/credentials.json | jq -r '."id-token"'`
    refresh_token=`cat $HOME/credentials.json | jq -r '."refresh-token"'`
    echo "key: $id_token" >> $HOME/.quantinuum_config
    echo "refresh: $refresh_token" >> $HOME/.quantinuum_config
    export CUDAQ_QUANTINUUM_CREDENTIALS=$HOME/.quantinuum_config


C++
````

For C++, the ``--target`` argument may be set to ``quantinuum``. ``nvq++`` will grab
the credentials from your home directory, authenticate them with the Quantinuum API,
and submit any quantum kernel executions to the hardware. By default, the QPU is set
to the Quantinuum syntax checker. This is helpful for determining the validity of your
kernel before submitting to a physical QPU. 

.. code:: bash

    nvq++ --target quantinuum src.cpp ...


Specifying the QPU
'''''''''''''''''''

The ``quantinuum`` target will select the Quantinuum syntax checker by default.
To specify a particular QPU, or "machine", this may be done in ``nvq++`` as

.. code:: bash

    nvq++ --target quantinuum --quantinuum-machine H1-2 src.cpp ...

where ``H1-2`` is an example of a physical QPU. Hardware specific
emulators may be accessed by appending an ``E`` to the end (e.g, ``H1-2E``). For 
access to the syntax checker for the provided machine, you may append an ``SC`` 
to the end (e.g, ``H1-1SC``).

For a comprehensive list of available machines, login to your `Quantinuum user account
<https://um.qapi.quantinuum.com/user>`_ and navigate to the "Account" tab, where you should
find a table titled "Machines".


Example
.......

.. literalinclude:: ../examples/cpp/quantinuum
    :language: cpp


Python 
```````

In python, the target may be controlled with the ``cudaq.set_target()``
function. This will set the target for any kernel executions within the file,
and will go through the same credential scheme as discussed in the C++ case.

.. code:: python

    cudaq.set_target('quantinuum')

To emulate the Quantinuum machine locally, without submitting through the cloud,
you can also set the ``emulate`` flag to ``True``. This will emit any target 
specific compiler warnings and diagnostics, before running a noise free emulation.

.. code:: python

    cudaq.set_target('quantinuum', emulate=True)

To select the number of shots for the kernel execution, this may be done through
the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
the ``shots_count`` is set to 1000.

.. code:: python 

    cudaq.sample(kernel, shots_count=10000)


Specifying the QPU
'''''''''''''''''''

In Python, specification of the QPU may be done by setting the :code:`machine` 
parameter of the target.

.. code:: python

    cudaq.set_target('quantinuum', machine='H1-2')

where ``H1-2`` is an example of a physical QPU. Hardware specific
emulators may be accessed by appending an ``E`` to the end (e.g, ``H1-2E``). For 
access to the syntax checker for the provided machine, you may append an ``SC`` 
to the end (e.g, ``H1-1SC``).

For a comprehensive list of available machines, login to your `Quantinuum user account
<https://um.qapi.quantinuum.com/user>`_ and navigate to the "Account" tab, where you should
find a table titled "Machines".


Example
.......

.. literalinclude:: ../examples/python/quantinuum
   :language: python
