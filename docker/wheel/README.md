# Pip Wheel Release Workflow

## Endpoint

The endpoint for the entire routine of building and testing the wheel is the
`build_and_test.sh` bash script.

This builds the manylinux image before building the CUDA Quantum pip wheel
within in. We run auditwheel to repair and rename the wheel, then pass it off to
the Dockerfile in `tests/Dockerfile.ubuntu2204` for testing. This image will
install the wheel, then pull down CUDA Quantum (doesn't build it), and runs the
python test suite to confirm the wheel behaves as expected.

## TODO

#### First PR

1. The audithweel repair call is failing due to an issue finding
`libdcudaq-spin.so`. The only thing that has changed since the last time it ran
properly was the removal of the STATIC_LIBZ and LIBZ_PATH arguments from the
cmake install. I have not had time to test if it works by reinserting them, but
that is my best guess at the moment for the issue.

2. I have the manylinux container cloning the wheel branch of my fork. Once the
first wheel PR is merged, that will need to be replaced.

3. The branch I'm writing this README from was originally branched off of my
wheel branch. That may cause a nightmare of issues with resolving commits, so I
may just port all of these changes to a new branch once the first PR is merged
into main.

#### Follow Up PR's

1. Particularly when it comes to integrating this with the CI, there may need to
be some refactoring on this wheel directory.

2. CI integration.

3. Uploading/distribution.
