# Welcome to the CUDA Quantum Python API

## Installing CUDA Quantum

To pull down the CUDA Quantum Python wheel distribution, simply run:

```console
pip install cuda-quantum
```

At this time, wheels are distributed for Linux operating systems only.

To build a source distribution of the Python API through pip:

```console
git clone https://github.com/NVIDIA/cuda-quantum.git
cd cuda-quantum
pip install . 
```

If you would like to build the entire C++ and Python API's from source,
follow the [installation instructions][official_install]. You
may then export your python path to the installation directory:

```console
export PYTHONPATH=$PYTHONPATH:/path/to/cudaq/install
```

[official_install]: https://nvidia.github.io/cuda-quantum/latest/install.html

## Running CUDA Quantum

You should now be able to import CUDA Quantum and start building quantum programs in
Python!

```console
import cudaq

kernel = cudaq.make_kernel()
qubit = kernel.qalloc()
kernel.x(qubit)
kernel.mz()

result = cudaq.sample(kernel)
```

## Documentation 

To see more examples, go to [python examples][python_examples], or check out the
[Python API reference][python_api_reference].

[python_examples]: https://nvidia.github.io/cuda-quantum/latest/using/python.html
[python_api_reference]: https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html

## Contributing

There are many ways in which you can get involved with CUDA Quantum. If you are
interested in developing quantum applications with CUDA Quantum, this repository
is a great place to get started! For more information about contributing to the
CUDA Quantum platform, please take a look at
[Contributing.md](../Contributing.md).

## License

The code in this repository is licensed under [Apache License 2.0](../LICENSE).

Contributing a pull request to this repository requires accepting the
Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. A CLA-bot will
automatically determine whether you need to provide a CLA and decorate the PR
appropriately. Simply follow the instructions provided by the bot. You will only
need to do this once.

## Feedback

Please let us know your feedback and ideas for the CUDA Quantum platform in the
[Discussions][cuda_quantum_discussions] tab of this repository, or file an
[issue][cuda_quantum_issues]. To report security concerns or [Code of
Conduct](../Code_of_Conduct.md) violations, please reach out to
[cuda-quantum@nvidia.com](mailto:cuda-quantum@nvidia.com).

[cuda_quantum_discussions]: https://github.com/NVIDIA/cuda-quantum/discussions
[cuda_quantum_issues]: https://github.com/NVIDIA/cuda-quantum/issues
