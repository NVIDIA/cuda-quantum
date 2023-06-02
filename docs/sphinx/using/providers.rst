CUDA Quantum Providers
*********************************

The QPU targets that are currently available in CUDA Quantum are as follows.

IonQ Quantum Cloud
==================================

To pass a compiled CUDA-quantum QIR circuit to IonQ, you set :code:`ionq` as the
target of your :code:`cudaq` object, as follows:

.. code:: bash 

    nvq++ --target ionq src.cpp ...

In Python, this can be specified with

.. code:: python 

    cudaq.set_target('ionq')

Export your `IonQ API key <https://cloud.ionq.com/>`_ to an environment variable,
:code:`IONQ_API_KEY`, to authenticate.

Targeting QPUs
++++++++++++++++++++++++++++++++++

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
