CUDA Quantum Providers
*********************************

The QPU targets that are currently available in CUDA Quantum are as follows.

IonQ Quantum Cloud
==================================

Simulator Backend

++++++++++++++++++++++++++++++++++

The :code:`ionq` backend passes the compiled cuda-quantum QIR circuits to IonQ. 

To specify the use of the :code:`ionq` backend, pass the following command line 
options to :code:`nvq++`

.. code:: bash 

    nvq++ --target ionq src.cpp ...

In Python, this can be specified with

.. code:: python 

    cudaq.set_target('ionq')

Export your `IonQ API key https://cloud.ionq.com/`_ to an environment variable, :code:`IONQ_API_KEY`, to authenticate.

Hardware Backend
++++++++++++++++++++++++++++++++++
