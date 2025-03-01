# Contributing

Thank you for your interest in contributing to CUDA-Q! Based on the type of
contribution, it will fall into three categories:

1. Report a bug, feature request, or documentation issue:

    File an [issue][cuda_quantum_issues] describing what you encountered or what
    you want to see changed. The NVIDIA team will evaluate the issues and triage
    them, scheduling them for a release. If you believe the issue needs priority
    attention comment on the issue to notify the team.

1. Share your work built upon CUDA-Q:

    We would love to hear more about your work! Please share with us on
    [NVIDIA/cudaq GitHub
    Discussions](https://github.com/NVIDIA/cuda-quantum/discussions) or consider
    contributing to our [examples](./docs/sphinx/examples/)! We also take any
    CUDA-Q related questions on this forum.

1. Implement a feature or bug-fix:

    Please file an [issue][cuda_quantum_issues] on the repository and express
    your interest in contributing to its implementation. Someone from the CUDA-Q
    team will respond on the issue to discuss how to best proceed with the
    suggestion. For more information regarding contributing to the code base,
    see also [Developing.md](./Developing.md).

[cuda_quantum_issues]: https://github.com/NVIDIA/cuda-quantum/issues

When you contribute code to this repository, whether be it an example, bug fix,
or feature, make sure that you can contribute your work under the used
[open-source license](./LICENSE), that is make sure no license and/or patent
conflict is introduced by your pull-request. To confirm this, you will need to
[sign off on your commits](#commit-sign-off) as described below. Thanks in advance
for your patience as we review your contributions; we do appreciate them!

## Commit Sign-off

We require that all contributors "sign-off" on their commits. This certifies
that the contribution is your original work, or you have rights to submit it
under the same license, or a compatible license. Any contribution which contains
commits that are not signed off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

```bash
git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```txt
Signed-off-by: Your Name <your@email.com>
```

By signing off on your commits you attest to the [Developer Certificate of Origin
(DCO)](https://developercertificate.org/). Full text of the DCO:

```txt
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
