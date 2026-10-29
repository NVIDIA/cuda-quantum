# CUDA-Q Community Projects

Each project listed on the CUDA-Q Community documentation page lives in its own
folder under `docs/sphinx/community/`. The page discovers projects
automatically, so listing a new one does not require editing any existing file.

## Listing a project

Add a folder `docs/sphinx/community/<project_name>/` containing:

- `index.rst`: the project's documentation page. It must begin with the field
  list below, followed by the page title.
- The example script referenced by the `:example:` field.

```rst
:repository: https://github.com/<org>/<project>
:maintainers: Jane Doe, John Roe
:license: Apache-2.0
:example: example.py
:summary: One-line description shown in the project list.

Project Title
=============
```

All five fields above are required. A project missing any of them, or a title,
is reported as a warning during the docs build and is left out of the list.

One optional field is also supported:

- `:install:` the arguments to `pip install` that the example needs before it
  runs, for example `:install: cudaq-mock-project==1.2`. Only package specifiers
  and pip options are supported, not arbitrary shell commands.

See [`mock_project/`](mock_project) for a complete, fictional example.
