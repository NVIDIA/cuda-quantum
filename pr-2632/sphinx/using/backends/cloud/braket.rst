Amazon Braket
++++++++++++++

.. _braket-backend:

`Amazon Braket <https://aws.amazon.com/braket/>`__ is a fully managed AWS 
service which provides Jupyter notebook environments, high-performance quantum 
circuit simulators, and secure, on-demand access to various quantum computers.
To get started users must enable Amazon Braket in their AWS account by following 
`these instructions <https://docs.aws.amazon.com/braket/latest/developerguide/braket-enable-overview.html>`__.
To learn more about Amazon Braket, you can view the `Amazon Braket Documentation <https://docs.aws.amazon.com/braket/>`__ 
and `Amazon Braket Examples <https://github.com/amazon-braket/amazon-braket-examples/tree/main/examples/nvidia_cuda_q>`__.
A list of available devices and regions can be found `here <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`__. 

Users can run CUDA-Q programs on Amazon Braket with `Hybrid Job <https://docs.aws.amazon.com/braket/latest/developerguide/braket-what-is-hybrid-job.html>`__.
See `this guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-first.html>`__ to get started with Hybrid Jobs and `this guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-using-cuda-q.html>`__ on how to use CUDA-Q with Amazon Braket.

Setting Credentials
```````````````````

After enabling Amazon Braket in AWS, set credentials using any of the documented `methods <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>`__.
One of the simplest ways is to use `AWS CLI <https://aws.amazon.com/cli/>`__.

.. code:: bash

    aws configure

Alternatively, users can set the following environment variables.

.. code:: bash

  export AWS_DEFAULT_REGION="<region>"
  export AWS_ACCESS_KEY_ID="<key_id>"
  export AWS_SECRET_ACCESS_KEY="<access_key>"
  export AWS_SESSION_TOKEN="<token>"

Submission from C++
`````````````````````````

To target quantum kernel code for execution in Amazon Braket,
pass the flag ``--target braket`` to the ``nvq++`` compiler.
By default jobs are submitted to the state vector simulator, `SV1`.

.. code:: bash

    nvq++ --target braket src.cpp

To execute your kernels on different device, pass the ``--braket-machine`` flag to the ``nvq++`` compiler
to specify which machine to submit quantum kernels to:

.. code:: bash

    nvq++ --target braket --braket-machine "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet" src.cpp ...

where ``arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet`` refers to IQM Garnet QPU.

To emulate the device locally, without submitting through the cloud,
you can also pass the ``--emulate`` flag to ``nvq++``. 

.. code:: bash

    nvq++ --emulate --target braket src.cpp

To see a complete example for using Amazon Braket backends, take a look at our :ref:`C++ examples <examples>`.

Submission from Python
`````````````````````````

The target to which quantum kernels are submitted 
can be controlled with the ``cudaq::set_target()`` function.

.. code:: python

    cudaq.set_target("braket")

By default, jobs are submitted to the state vector simulator, `SV1`.

To specify which Amazon Braket device to use, set the :code:`machine` parameter.

.. code:: python

    device_arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
    cudaq.set_target("braket", machine=device_arn)

where ``arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet`` refers to IQM Garnet QPU.

To emulate the device locally, without submitting through the cloud,
you can also set the ``emulate`` flag to ``True``.

.. code:: python

    cudaq.set_target("braket", emulate=True)

The number of shots for a kernel execution can be set through the ``shots_count``
argument to ``cudaq.sample``. By default, the ``shots_count`` is set to 1000.

.. code:: python

    cudaq.sample(kernel, shots_count=100)

To see a complete example for using Amazon Braket backends, take a look at our :ref:`Python examples <examples>`.

.. note:: 

    The ``cudaq.observe`` API is not yet supported on the `braket` target.
