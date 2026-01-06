#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Release process

# To see all the changes that were added since the latest release, navigate to your clone of the cuda-quantum repository, 
# and run this script:

version_regex="([0-9]{1,}\.)+[0-9]{1,}\S*"
versions=`gh release list -R nvidia/cuda-quantum --exclude-drafts --exclude-pre-releases | egrep -o "$version_regex" | sort -r -V`
last_release=`echo $versions | cut -d ' ' -f 1`
current_release="$1"
if [ -z "$current_release" ]; then 
    echo "Warning: no version number specified for the new release."
    rel_branch=main
else
    rel_branch=releases/v$current_release
fi

author="" # insert your name to show only your PRs
git pull || echo "Warning: failed to pull updates"
git log $rel_branch...releases/v$last_release --cherry-pick --left-only --no-merges --oneline --author="$author" | egrep -o '\(#[0-9]*\)$' > commits.txt

for pr in `cat commits.txt`; do 
    maintenance=`gh pr view ${pr: 2: -1} --json labels --jq 'any(.labels.[]; .name == "maintenance")'`
    if ! $maintenance; then
        milestone=`gh pr view ${pr: 2: -1} --json milestone --jq '.milestone.title'`
        pr_author=`gh pr view ${pr: 2: -1} --json author --jq '.author.name'`

        if [ -z "$milestone" ]; then
            echo "Missing milestone for PR ${pr: 2: -1} by $pr_author."
        elif [ -z "$current_release" ] || [ "$(echo "$milestone" | egrep -o "$version_regex" || true)" == "$current_release" ]; then
            labels=`gh pr view ${pr: 2: -1} --json labels --jq '.labels.[].name'`
            echo "Labels for PR ${pr: 2: -1} by $pr_author: $labels"
        fi
    fi
done

# Make sure all listed PRs are labeled appropriately.

# Manually launch the deployment workflow from the release branch, setting the appropriate version number for the GitHub release.
# -> this should automatically launch the publishing workflow from the main branch when the deployment completes
# -> this should automatically launch the documentation publishing for that version
# Check that all three workflows above completed successfully. 
# Once the publishing completes, you should see a draft release on GitHub for the new version.

# Check that all nightly integration tests are enabled and run successfully with the release image. 
# Work with QA to get the release candidate fully validated.

# Go to the draft release on GitHub and download the python wheels and metapackages in the draft release.
# Note: every wheel for a given version can only be uploaded once to pypi or test-pypi. 
# Hence, test with a single wheel at a time to allow to check any needed fixes with other wheels.
# Upload a suitable python wheel to test-pypi and check that the project site (generated based on the README in the wheel) looks good. 

# If necessary, perform additional manual validation of the wheels. This is necessary, for example, to test external backends
# that are disabled in our pipelines (e.g. due to submission costs).
# To do so following these steps:
# - login to a multi-GPU system, start a clean environment for testing in the form of a docker container (you could look at the docker/test/*.Dockerfile on our repo to use as a starting point)
# - install the wheel you uploaded via pip passing the argument `--extra-index-url https://test.pypi.org/simple/`; this may fail with a cuquantum not found, if the cuquantum dependency is not yet released
# - if needed install all unreleased dependencies listed in the pyproject.toml file manually, and then install the cuda-quantum wheel; this should now work
# - follow the instruction on the project site and our docs to setup you environment (both with and without using conda)
# - check that you can use all features that require manual validation
# Upload the cudaq "metapackage" (Python sdist) to test-pypi and check that the readme looks good.

# If everything looks good, upload the python wheels and Python metapackages to PyPI - 
# note that we cannot update that version of the wheels once they have been uploaded!

# Look at the draft release created by the publishing workflow and update the release notes manually as needed:
# - Update the link to the full change log to show the diff between the tag of the current release and the previous one;
#   the link will not work until the release is made public since the tag does not yet exit
# - Delete the Python wheels and the Python metapackages from the draft release.
# - Add a sentence or two summarizing the most important features in that release, and add links to the appropriate docs pages.
# Once everything looks good, make the release public. This will create a snapshot on Zanado, and we cannot update the release once it is public!

# Update the readme on NGC:
# - Sign in with the appropriate credentials and select the nvidia org and the quantum team.
# - Select the Private Registry, go to containers, and search for the CUDA-Q container.
# - Click on edit details and update the readme as needed; make sure to update the tags in the install commands.
# - Click save and confirm that changes.

# Publish the release image on the nightly channel under the matching tag in stable releases by manually launching the workflow publish_stable.yml
# Update the releases.rst file on the main branch to add the new release.
