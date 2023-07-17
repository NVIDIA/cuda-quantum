CUDA Quantum Hardware Backends
*********************************

The hardware vendors currently available in CUDA Quantum are as follows.

IonQ
==================================

Setting Credentials
###################

To execute programs on IonQ hardware from either C++ or Python, CUDA Quantum
will look for an API key stored in the `IONQ_API_KEY` environment variable. 
This may be set as follows, replacing the string with the API key generated
from your `IonQ account <https://cloud.ionq.com/>`_.

.. code:: bash

  export IONQ_API_KEY="ionq_generated_api_key"

For C++, it's straightforward to control the target QPU via the `--target`
argument to `nvq++`. This will look for the `IONQ_API_KEY` in your environment,
authenticate it with the IonQ API, and submit any quantum kernel executions within
the file to the hardware.

.. code:: bash 

    nvq++ --target ionq src.cpp ...

In python, the target may be controlled with the `cudaq.set_target()` [TODO: LINK]
function. This will set the target for any kernel executions within the file, 
and will go through the same credential scheme as discussed in the C++ case. 

.. code:: python 

    cudaq.set_target('ionq')

Specifying the QPU
###################

By default, the IonQ target will use the :code:`simulator` QPU.
To specify which IonQ QPU to use, set the :code:`qpu` parameter.
A list of available QPUs can be found `in the API documentation <https://docs.ionq.com/#tag/jobs>`_.

.. code:: c

    auto &platform = cudaq::get_platform();
    platform.setTargetBackend("ionq;qpu;qpu.aria-1");

.. code:: python
    cudaq.set_target("ionq", qpu="qpu.aria-1")

Note: A "target" in :code:`cudaq` refers to a quantum compute provider, such as :code:`ionq`.
However, IonQ's docs use the term "target" to refer to specific QPUs themselves.




Quantinuum 
==================================

Setting Credentials
###################

To execute programs on Quantinuum hardware from either C++ or Python, CUDA Quantum 
will look for a credentials file stored in your home directory. This file
may be generated with the following script, replacing the email and 
password with your Quantinuum login credentials.

.. code:: bash 
  # May need to install: `apt-get update && apt-get install curl jq`
  curl -X POST -H "Content Type: application/json" -d '{ "email":"<your_alias>@nvidia.com","password":"<your_password>" }' https://qapi.quantinuum.com/v1/login > $HOME/credentials.json
  id_token=`cat $HOME/credentials.json | jq -r '."id-token"'`
  refresh_token=`cat $HOME/credentials.json | jq -r '."refresh-token"'`
  echo "key: $id_token" >> $HOME/.quantinuum_config
  echo "time: 0" >> $HOME/.quantinuum_config
  echo "refresh: $refresh_token" >> $HOME/.quantinuum_config
  export CUDAQ_QUANTINUUM_CREDENTIALS=$HOME/.quantinuum_config

For C++, the `--target` argument may be set to "quantinuum". `nvq++` will grab 
the credentials from your home directory, authenticate them with the Quantinuum API, 
and submit any quantum kernel executions to the hardware.

.. code:: bash 

    nvq++ --target quantinuum src.cpp ...

In python, the target may be controlled with the `cudaq.set_target()` [TODO: LINK]
function. This will set the target for any kernel executions within the file, 
and will go through the same credential scheme as discussed in the C++ case. 

.. code:: python 

    cudaq.set_target('quantinuum')

For full examples in C++, see here [TODO: LINK], and here [TODO: LINK] for Python.