# Welcome to the CUDA Quantum repository

<img align="right" width="200"
src="https://developer.nvidia.com/sites/default/files/akamai/nvidia-cuquantum-icon.svg"
/>

<img align="left"
src="https://github.com/NVIDIA/cuda-quantum/actions/workflows/deployments.yml/badge.svg?event=workflow_dispatch&branch=main"
/>

<img align="left"
src="https://github.com/NVIDIA/cuda-quantum/actions/workflows/publishing.yml/badge.svg?branch=main"
/>

<img align="left"
src="https://github.com/NVIDIA/cuda-quantum/actions/workflows/documentation.yml/badge.svg?branch=main"
/> <br/>

<a href="https://zenodo.org/badge/latestdoi/614026597"><img align="left"
src="https://zenodo.org/badge/614026597.svg" alt="DOI"></a><br/>

The CUDA Quantum Platform for hybrid quantum-classical computers enables
integration and programming of quantum processing units (QPUs), GPUs, and CPUs
in one system. This repository contains the source code for all C++ and Python
tools provided by the CUDA Quantum toolkit, including the `nvq++` compiler, the
CUDA Quantum runtime, as well as a selection of integrated CPU and GPU backends
for rapid application development and testing.

## Getting Started

To learn more about how to work with CUDA Quantum, please take a look at the
[CUDA Quantum Documentation][cuda_quantum_docs]. The page also contains
[installation instructions][official_install] for officially released packages.

If you would like to install the latest iteration under development in this
repository and/or add your own modifications, take a look at the [latest
packages][github_packages] deployed on the GitHub Container Registry. For more
information about building CUDA Quantum from source, see [these
instructions](./Building.md).

[cuda_quantum_docs]: https://nvidia.github.io/cuda-quantum/latest
[official_install]: https://nvidia.github.io/cuda-quantum/latest/install.html
[github_packages]:
    https://github.com/orgs/NVIDIA/packages?repo_name=cuda-quantum

## Contributing

There are many ways in which you can get involved with CUDA Quantum. If you are
interested in developing quantum applications with CUDA Quantum, this repository
is a great place to get started! For more information about contributing to the
CUDA Quantum platform, please take a look at
[Contributing.md](./Contributing.md).

## License

The code in this repository is licensed under [Apache License 2.0](./LICENSE).

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
Conduct](./Code_of_Conduct.md) violations, please reach out to
[cuda-quantum@nvidia.com](mailto:cuda-quantum@nvidia.com).

[cuda_quantum_discussions]: https://github.com/NVIDIA/cuda-quantum/discussions
[cuda_quantum_issues]: https://github.com/NVIDIA/cuda-quantum/issues
