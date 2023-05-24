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

    nvq++ --qpu ionq src.cpp ...

In Python, this can be specified with 

.. code:: python 

    cudaq.set_qpu('ionq')

Hardware Backend
++++++++++++++++++++++++++++++++++


