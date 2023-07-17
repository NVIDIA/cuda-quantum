CUDA Quantum Hardware Backends
*********************************

The hardware vendors currently available in CUDA Quantum are as follows.

IonQ
==================================
Words...

.. code:: bash 

    nvq++ --target ionq src.cpp ...

In python, this can be specified with 

.. code:: python 

    cudaq.set_target('ionq')



Quantinuum 
==================================

### Setting Credentials
To execute programs on Quantinuum from either C++ or Python, CUDA Quantum 
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

For C++, it's straightforward to control the target QPU via the `--target`
argument to `nvq++`. This will grab the credentials from your home directory,
validate them with the Quantinuum API, and submit any quantum kernel executions
to the hardware.

.. code:: bash 

    nvq++ --target quantinuum src.cpp ...

In python, the target may be controlled with the `cudaq.set_target()` [TODO: LINK]
function. This will set the target for any kernel executions within the file, 
and will go through the same credential scheme as discussed in the C++ case. 

.. code:: python 

    cudaq.set_target('quantinuum')

For full examples in C++, see here [TODO: LINK], and here [TODO: LINK] for Python.