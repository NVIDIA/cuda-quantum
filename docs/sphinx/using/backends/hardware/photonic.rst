Photonic
==========

ORCA Computing
+++++++++++++++

.. _orca-backend:

ORCA Computing's PT Series implement the boson sampling model of quantum computation, in which 
multiple single photons are interfered with each other within a network of beam splitters, and 
photon detectors measure where the photons leave this network. This process is implemented within 
a time-bin interferometer (TBI) architecture where photons are created in different time-bins 
and interfered within a network of delay lines. This can be represented by a circuit diagram, 
like the one below, where this illustration example corresponds to 4 photons in 8 modes sent into 
alternating time-bins in a circuit composed of two delay lines in series.

.. image:: ../../examples/images/orca_tbi.png
   :width: 400px
   :align: center


Setting Credentials
```````````````````

Programmers of CUDA-Q may access the ORCA API from either C++ or Python. There is an environment 
variable ``ORCA_ACCESS_URL`` that can be set so that the ORCA target can look for it during 
configuration.

.. code:: bash

  |:spellcheck-disable:|export ORCA_ACCESS_URL="https://<ORCA API Server>"|:spellcheck-enable:|


Sometimes the requests to the PT-1 require an authentication token. This token can be set as an
environment variable named ``ORCA_AUTH_TOKEN``. For example, if the token is :code:`AbCdEf123456`,
you can set the environment variable as follows:

.. code:: bash

  |:spellcheck-disable:|export ORCA_AUTH_TOKEN="AbCdEf123456"|:spellcheck-enable:|

Submitting
`````````````````````````

.. tab:: Python

        To set which ORCA URL to be used, set the :code:`url` parameter.

        .. code:: python

            import os
            import cudaq
            # ...
            orca_url = os.getenv("ORCA_ACCESS_URL", "http://localhost/sample")

            cudaq.set_target("orca", url=orca_url)


        You can then execute a time-bin boson sampling experiment against the platform using an ORCA device.

        .. code:: python

            bs_angles = [np.pi / 3, np.pi / 6]
            input_state = [1, 1, 1]
            loop_lengths = [1]
            counts = cudaq.orca.sample(input_state, loop_lengths, bs_angles)

        To see a complete example for using ORCA's backends, take a look at our :doc:`Python examples <../../examples/hardware_providers>`.



.. tab:: C++

        
        To execute a boson sampling experiment on the ORCA platform, provide the flag 
        ``--target orca`` to the ``nvq++`` compiler. You should then pass the ``--orca-url`` flag set with 
        the previously set environment variable ``$ORCA_ACCESS_URL`` or an :code:`url`.

        .. code:: bash

            nvq++ --target orca --orca-url $ORCA_ACCESS_URL src.cpp -o executable

        or

        .. code:: bash

            nvq++ --target orca --orca-url <url> src.cpp -o executable

        To run the output, invoke the executable

        .. code:: bash

           ./executable


        To see a complete example for using ORCA server backends, take a look at our :doc:`C++ examples <../../examples/hardware_providers>`.
