CUDA Quantum Hardware Backends
*********************************

The hardware vendors currently available in CUDA Quantum are as follows.

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

At this time, programmer control over the IonQ QPU is under construction.


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

Note: A "target" in :code:`cudaq` refers to a quantum compute provider, such as :code:`ionq`.
However, IonQ's documentation uses the term "target" to refer to specific QPU's themselves.

Specifying the QPU
'''''''''''''''''''

At this time, programmer control over the IonQ QPU is under construction.

.. By default, the IonQ target will use the :code:`simulator` QPU.
.. To specify which IonQ QPU to use, set the :code:`qpu` parameter.
.. A list of available QPUs can be found `in the API documentation
.. <https://docs.ionq.com/#tag/jobs>`_.

.. .. code:: python
..     cudaq.set_target("ionq", qpu="qpu.aria-1")

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
  echo "time: 0" >> $HOME/.quantinuum_config
  echo "refresh: $refresh_token" >> $HOME/.quantinuum_config
  export CUDAQ_QUANTINUUM_CREDENTIALS=$HOME/.quantinuum_config


C++
````

For C++, the ``--target`` argument may be set to "quantinuum". ``nvq++`` will grab
the credentials from your home directory, authenticate them with the Quantinuum API,
and submit any quantum kernel executions to the hardware.

.. code:: bash

    nvq++ --target quantinuum src.cpp ...


Specifying the QPU
'''''''''''''''''''

The ``quantinuum`` target will select a Quantinuum emulator by default.
To specify a different QPU, this may be done in ``nvq++`` as

.. code:: bash

    nvq++ --target quantinuum -quantinuum-machine H1-2 src.cpp ...

where ``H1-2`` is an example of a physical QPU. Hardware specific
emulators may be accessed by appending an "E" to the end (e.g, ``H1-2E``).


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

Specifying the QPU
'''''''''''''''''''

At this time, control over the Quantinuum QPU for Python is under construction. 

.. In python, this may be done by setting the :code:`machine` parameter.

.. .. code:: python

..     cudaq.set_target('quantinuum', machine='H1-2')


Example
.......

.. literalinclude:: ../examples/python/quantinuum
   :language: python
