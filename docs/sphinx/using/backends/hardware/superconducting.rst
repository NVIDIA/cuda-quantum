Superconducting
=================

Anyon Technologies/Anyon Computing
+++++++++++++++++++++++++++++++++++

.. _anyon-backend:

Setting Credentials
```````````````````

Programmers of CUDA-Q may access the Anyon API from either
C++ or Python. Anyon requires a credential configuration file with username and password. 
The configuration file can be generated as follows, replacing
the ``<username>`` and ``<password>`` in the first line with your Anyon Technologies
account details. The credential in the file will be used by CUDA-Q to login to Anyon quantum services 
and will be updated by CUDA-Q with an obtained API token and refresh token. 
Note, the credential line will be deleted in the updated configuration file. 

.. code:: bash
    
    echo 'credentials: {"username":"<username>","password":"<password>"}' > $HOME/.anyon_config

Users can also login and get the keys manually using the following commands:

.. code:: bash

    # You may need to run: `apt-get update && apt-get install curl jq`
    curl -X POST --user "<username>:<password>"  -H "Content-Type: application/json" \
    https://api.anyon.cloud:5000/login > credentials.json
    id_token=`cat credentials.json | jq -r '."id_token"'`
    refresh_token=`cat credentials.json | jq -r '."refresh_token"'`
    echo "key: $id_token" > ~/.anyon_config
    echo "refresh: $refresh_token" >> ~/.anyon_config

The path to the configuration can be specified as an environment variable:

.. code:: bash

    export CUDAQ_ANYON_CREDENTIALS=$HOME/.anyon_config

Submitting
```````````````````

.. tab:: Python


        The target to which quantum kernels are submitted 
        can be controlled with the ``cudaq.set_target()`` function.

        To execute your kernels using Anyon Technologies backends, specify which machine to submit quantum kernels to
        by setting the :code:`machine` parameter of the target. 
        If :code:`machine` is not specified, the default machine will be ``telegraph-8q``.

        .. code:: python

            cudaq.set_target('anyon', machine='telegraph-8q')

        As shown above, ``telegraph-8q`` is an example of a physical QPU.

        To emulate the Anyon Technologies machine locally, without submitting through the cloud,
        you can also set the ``emulate`` flag to ``True``. This will emit any target 
        specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: python

            cudaq.set_target('anyon', emulate=True)

        The number of shots for a kernel execution can be set through
        the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
        the ``shots_count`` is set to 1000.

        .. code:: python 

            cudaq.sample(kernel, shots_count=10000)


.. tab:: C++


        To target quantum kernel code for execution in the Anyon Technologies backends,
        pass the flag ``--target anyon`` to the ``nvq++`` compiler. CUDA-Q will 
        authenticate via the Anyon Technologies REST API using the credential in your configuration file.

        .. code:: bash

            nvq++ --target anyon --<backend-type> <machine> src.cpp ...

        To execute your kernels using Anyon Technologies backends, pass the ``--anyon-machine`` flag to the ``nvq++`` compiler
        as the ``--<backend-type>`` to specify which machine to submit quantum kernels to:

        .. code:: bash

            nvq++ --target anyon --anyon-machine telegraph-8q src.cpp ...

        where ``telegraph-8q`` is an example of a physical QPU (Architecture: Telegraph, Qubit Count: 8).

        Currently, ``telegraph-8q`` and ``berkeley-25q`` are available for access over CUDA-Q.

        To emulate the Anyon Technologies machine locally, without submitting through the cloud,
        you can also pass the ``--emulate`` flag as the ``--<backend-type>`` to ``nvq++``. This will emit any target 
        specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: bash

            nvq++ --target anyon --emulate src.cpp

To see a complete example, take a look at :ref:`Anyon examples <anyon-examples>`.


IQM
+++

.. _iqm-backend:

`IQM Resonance <https://meetiqm.com/products/iqm-resonance/>`__ offers access to various different IQM quantum computers.
The machines available there will be constantly extended as development progresses.
Programmers of CUDA-Q may use IQM Resonance with either C++ or Python.

With this version it is no longer necessary to define the target QPU architecture in the code or at compile time.
The IQM backend integration now contacts at runtime the configured IQM server and fetches the active dynamic quantum architecture of the QPU.
This is then used as input to transpile the quantum kernel code just-in-time for the target QPU topology.
By setting the environment variable ``IQM_SERVER_URL`` the target server can be selected just before executing the program.
As result the python script or the compiled C++ program can be executed on different QPUs without recompilation or code changes.

Please find also more documentation after logging in to the IQM Resonance portal.


Setting Credentials
```````````````````

Create a free account on the `IQM Resonance portal <https://meetiqm.com/products/iqm-resonance/>`__ and log-in.
Navigate to the account profile (top right). There generate an "API Token" and copy the generated token-string.
Set the environment variable ``IQM_TOKEN`` to contain the value of the token-string.
The IQM backend integration will use this as authorization token at the IQM server.


Submitting
``````````

.. tab:: Python

    The target to which quantum kernels are submitted can be controlled with the ``cudaq.set_target()`` function.

    .. code:: python

        cudaq.set_target("iqm", url="https://<IQM Server>/")

    Please note that setting the environment variable ``IQM_SERVER_URL`` takes precedence over the URL configured in the code.


.. tab:: C++

    To target quantum kernel code for execution on an IQM Server, pass the ``--target iqm`` option to the ``nvq++`` compiler.

    .. code:: bash

        nvq++ --target iqm src.cpp

    Once the binary for an IQM QPU is compiled, it can be executed against any IQM Server by setting the environment variable ``IQM_SERVER_URL`` as shown here:

    .. code:: bash

        nvq++ --target iqm src.cpp -o program
        IQM_SERVER_URL="https://demo.qc.iqm.fi/" ./program


To see a complete example for using IQM server backends, take a look at :ref:`IQM examples <iqm-examples>`.


Advanced use cases
``````````````````

The IQM backend integration offers more options for advanced use cases. Please find these here:

.. toctree::
   :maxdepth: 2

        IQM backend advanced use cases <backend_iqm.rst>


OQC
++++

.. _oqc-backend:


`Oxford Quantum Circuits <https://oxfordquantumcircuits.com/>`__ (OQC) is currently providing CUDA-Q integration for multiple Quantum Processing Unit types.
The 8 qubit ring topology Lucy device and the 32 qubit Kagome lattice topology Toshiko device are both supported via machine options described below.

Setting Credentials
`````````````````````````

In order to use the OQC devices you will need to register.
Registration is achieved by contacting `oqc_qcaas_support@oxfordquantumcircuits.com`.

Once registered you will be able to authenticate with your ``email`` and ``password``

There are three environment variables that the OQC target will look for during configuration:

1. ``OQC_URL``
2. ``OQC_EMAIL``
3. ``OQC_PASSWORD`` - is mandatory


Submitting
`````````````````````````

.. tab:: Python

        To set which OQC URL, set the :code:`url` parameter.
        To set which OQC email, set the :code:`email` parameter.
        To set which OQC machine, set the :code:`machine` parameter.

        .. code:: python

            import os
            import cudaq
            os.environ['OQC_PASSWORD'] = password
            cudaq.set_target("oqc", url=url, machine="lucy")

        You can then execute a kernel against the platform using the OQC Lucy device

        To emulate the OQC device locally, without submitting through the OQC QCaaS services, you can set the ``emulate`` flag to ``True``.
        This will emit any target specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: python

            cudaq.set_target("oqc", emulate=True)


.. tab:: C++


        To target quantum kernel code for execution on the OQC platform, provide the flag ``--target oqc`` to the ``nvq++`` compiler.

        Users may provide their :code:`email` and :code:`url` as extra arguments

        .. code:: bash

            nvq++ --target oqc --oqc-email <email> --oqc-url <url> src.cpp -o executable

        Where both environment variables and extra arguments are supplied, precedent is given to the extra arguments.
        To run the output, provide the runtime loaded variables and invoke the pre-built executable

        .. code:: bash

           OQC_PASSWORD=<password> ./executable

        To emulate the OQC device locally, without submitting through the OQC QCaaS services, you can pass the ``--emulate`` flag to ``nvq++``.
        This will emit any target specific compiler warnings and diagnostics, before running a noise free emulation.

        .. code:: bash

            nvq++ --emulate --target oqc src.cpp -o executable

        .. note::

            The oqc target supports a ``--oqc-machine`` option.
            The default is the 8 qubit Lucy device.
            You can set this to be either ``toshiko`` or ``lucy`` via this flag.

.. note::

    The OQC quantum assembly toolchain (qat) which is used to compile and execute instructions can be found on github as `oqc-community/qat <https://github.com/oqc-community/qat>`__

To see a complete example, take a look at :ref:`OQC examples <oqc-examples>`.


Quantum Circuits, Inc.
+++++++++++++++++++++++

.. _qci-backend:

Quantum Circuits offers users the ability to execute CUDA-Q programs on its 
`Seeker QPU <https://quantumcircuits.com/product/#seeker>`__ and simulate 
them using its simulator, `AquSim <https://quantumcircuits.com/product/#simulator>`__. 
The Seeker is the first dual-rail qubit QPU available over the cloud today, and through
CUDA-Q users have access to its universal gate set, high fidelity operations, and fast 
throughput. Upcoming releases of CUDA-Q will continue to evolve these capabilities to 
include real-time control flow and access to an expanded collection of actionable data 
enabled by the Quantum Circuits error aware technology.

AquSim models error detection and real-time control of Quantum Circuits’ Dual-Rail Cavity Qubit 
systems, and uses a Monte Carlo approach to do so on a shot-by-shot basis. The supported 
features include all of the single and two-qubit gates offered by CUDA-Q. AquSim additionally 
supports real-time conditional logic enabled by feed-forward capability. Noise modeling is 
offered, effectively enabling users to emulate the execution of programs on the Seeker QPU 
and thereby providing a powerful application prototyping tool to be leveraged in advance of 
execution on hardware.

With C++ and Python programming supported, users are able to prototype, test and explore 
quantum applications in CUDA-Q on the Seeker and AquSim. Users who wish to get started with 
running CUDA-Q with Quantum Circuits should visit our `Explore <https://quantumcircuits.com/explore/>`__ 
page to learn more about the Quantum Circuits Select Quantum Release Program.

Installation & Getting Started
```````````````````````````````

.. |:spellcheck-disable:| replace:: \
.. |:spellcheck-enable:| replace:: \

Until CUDA-Q release 0.13.0 is available, the integration with Quantum Circuits will be supported through the 
|:spellcheck-disable:| `nightly build Docker images <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nightly/containers/cuda-quantum/tags>`__. |:spellcheck-enable:|

Instructions on how to install and get started with CUDA-Q using Docker can be found :ref:`here <install-docker-image>`.

You may present your user token to Quantum Circuits via CUDA-Q by setting an environment variable 
named :code:`QCI_AUTH_TOKEN` before running your CUDA-Q program. 

For example:

.. code:: bash

    export QCI_AUTH_TOKEN="example-token"

Tokens are provided as part of the Strategic Quantum Release Program. Please visit our 
`Explore <https://quantumcircuits.com/explore/>`__  page to learn more.

Using CUDA-Q with Quantum Circuits
```````````````````````````````````

Quantum Circuits' Seeker system detects errors in real-time and returns not just 0s and 1s 
as the measurement outcomes, but unique results tagged as -1, which indicate that an erasure 
was detected on the Dual-Rail Cavity Qubit. AquSim emulates this execution as well, enabling 
users to model error aware programs in advance of execution on the QPU. While -1 data is not 
yet available via the CUDA-Q API, the user still has insight into these dynamics through the 
number of shots that are collected in a given run.


Yield
```````

Quantum Circuits architecture can detect errors in measurements. The target will return to
the user the outcome from every measurement for every shot, regardless of
whether errors were detected. However, the data from a shot in which any of the
measurements had an error detected will:

- Every **RESULT** where an error is detected will be ``-1`` (instead of ``0``
  or ``1``).
- The shot will be marked with an **exit code** of ``1`` (instead of ``0``).
- It will be **excluded** from the histogram.

Apart from an ideal simulation, most jobs will include at least some shots for
which errors were detected.

The shots that have no errors detected are referred to as **post-selected** and
will have an exit code of ``0``. The **yield** represents the fraction of
executed shots that are not rejected due to detected errors:

.. math::

    \text{yield} = \frac{\text{number of post-selected shots}}{\text{number of shots executed}}

The yield depends on the number of qubits and the depth of the circuit.

Options
`````````

**machine**
    This is a string option with 2 supported values.

    - **Seeker**

      - Name of the QPU supported by Quantum Circuits.
      - Supports up to **8 qubit** programs and the ``base_profile``.
      - Regardless of whether the method is ``execute`` or ``simulate``, the
        program will be **fully compiled** for strict validation of suitability
        to run on the QPU.

    - **AquSim**

      - This "machine" is not associated with a specific QPU and not strictly
        validated.
      - Supports up to **25 qubits**, a **square grid coupling map**, and the
        ``adaptive_profile``.

**method**
    This is a string option with 2 supported values.

    - **execute**

      - If ``machine="Seeker"``, the program will run on the QPU (depending on
        availability).
      - Not supported if ``machine="AquSim"``.

    - **simulate**

      - The program will be run in ``AquSim``.

**noisy**
    This boolean option is only supported for ``method="simulate"``.

    - **True**

      - ``AquSim`` will simulate noise and error detection using a **Dual-Rail
        statevector-based noise model** on a transpiled program.

    - **False**

      - An **ideal simulation**.

**repeat_until_shots_requested**
    This is a boolean option.

    - **True**

      - The machine will return as many post-selected shots as were requested
        (unless an upper limit of shots executed is encountered first).
      - The **execution time is proportional to 1 / yield**.

    - **False**

      - The machine will execute **exactly the number of shots requested**,
        regardless of how many errors are detected.
      - The execution time does **not depend on yield**.


Submitting
```````````

.. tab:: Python

        To set the target to Quantum Circuits, add the following to your Python
        program:

        .. code:: python

            cudaq.set_target('qci')
            [... your Python here]

        To run on AquSim, simply execute the script using your Python interpreter.
        
        To specify which Quantum Circuits machine to use, set the :code:`machine` parameter:

        .. code:: python

            # The default machine is AquSim
            cudaq.set_target('qci', machine='AquSim') 
            # or
            cudaq.set_target('qci', machine='Seeker')

        You can control the execution method using the :code:`method` parameter:

        .. code:: python

            # For simulation (default)
            cudaq.set_target('Seeker', method='simulate')
            # For hardware execution
            cudaq.set_target('Seeker', method='execute')

        For noisy simulation, you can enable the :code:`noisy` parameter:

        .. code:: python

            cudaq.set_target('qci', noisy=True)

        When collecting shots, you can ensure the requested number of shots are obtained
        by enabling the :code:`repeat_until_shots_requested` parameter:

        .. code:: python

            cudaq.set_target('qci', repeat_until_shots_requested=True)


.. tab:: C++

        When executing programs in C++, they must first be compiled using the
        CUDA-Q nvq++ compiler, and then submitted to run on the Seeker or AquSim.

        Note that your token is fetched from your environment at run time, not at compile time.

        In the example below, the compilation step shows two flags being passed to the nvq++
        compiler: the Quantum Circuits target :code:`--target qci`, and the output file
        :code:`-o example.x`.  The second line executes the program against AquSim. Here are the
        shell commands in full:

        .. code:: bash

            nvq++ example.cpp --target qci -o example.x
            ./example.x

        To specify which Quantum Circuits machine to use, pass the ``--qci-machine`` flag:

        .. code:: bash

            # The default machine is AquSim
            nvq++ --target qci --qci-machine AquSim src.cpp -o example.x
            # or
            nvq++ --target qci --qci-machine Seeker src.cpp -o example.x

        You can control the execution method using the ``--qci-method`` flag:

        .. code:: bash

            # For simulation (default)
            nvq++ --target qci --qci-machine Seeker --qci-method simulate src.cpp -o example.x
            # For hardware execution
            nvq++ --target qci --qci-machine Seeker --qci-method execute src.cpp -o example.x

        For noisy simulation, you can set the ``--qci-noisy`` argument to `true`:

        .. code:: bash

            nvq++ --target qci --qci-noisy true src.cpp -o example.x

        When collecting shots, you can ensure the requested number of shots are obtained
        with the ``--qci-repeat_until_shots_requested`` argument:

        .. code:: bash

            nvq++ --target qci --qci-repeat_until_shots_requested true src.cpp -o example.x

.. note::
    By default, only successful shots are presented to the user and may be fewer than the 
    requested number. Enabling :code:`repeat_until_shots_requested` ensures the full 
    requested shot count is collected, at the cost of increased execution time.


To see a complete example of using Quantum Circuits' backends, please take a look at the
:ref:`Quantum Circuits examples <quantum-circuits-examples>`.

.. note:: 

        In local emulation mode (``emulate`` flag set to ``True``), the program will be executed on the :ref:`default simulator <default-simulator>`.
        The environment variable ``CUDAQ_DEFAULT_SIMULATOR`` can be used to change the emulation simulator. 
        
        For example, the simulation floating point accuracy and/or the simulation capabilities (e.g., maximum number of qubits, supported quantum gates),
        depend on the selected simulator.  
        
        Any environment variables must be set prior to setting the target or running "`import cudaq`".
