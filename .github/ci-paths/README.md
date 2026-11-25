# Path gates for CI

The `CI` and `CI - macOS` workflows skip jobs that a change cannot affect. A
`changes` job runs `.github/actions/check-changed-paths` once per gate and
publishes the results. Each gate passes when the pull request touches at least
one path that is not on the gate's ignore lists. Everything else fails open:
an unknown path, an API error or a missing list runs the jobs.

## Scope

Only pull request pushes (`pull-request/N` branches) are gated. The merge queue,
`workflow_dispatch`, `schedule` and the release workflows always run every job,
so the merge queue and the daily run catch a gate that is wrong.

## Path lists

| File | Contents |
| --- | --- |
| `inert.txt` | Repository metadata that no workflow reads. Skips the whole workflow. |
| `docs-prose.txt` | `*.md` and `*.rst`. Needs the documentation build only. |
| `logical-only.txt` | `preview/` and `edem/`. Not read by the core build, tests or docs. |
| `not-linux-ci.txt` | Workflows and actions that `ci.yml` never calls. Generated. |
| `not-macos-ci.txt` | Workflows and actions that `ci_macos.yml` never calls, plus Dockerfiles. Generated. |

An entry is a path prefix, or a glob when it contains `*`, `?` or `[`.
Editing any list file runs everything, because the action treats a change to a
list as a change to the gate. The two generated lists come from
`workflow_scopes.py`:

```bash
python3 .github/ci-paths/workflow_scopes.py --write
python3 .github/ci-paths/workflow_scopes.py --check
```

## Gates

| Gate | Ignores | Controls |
| --- | --- | --- |
| `stable` | research previews, `inert.txt` | the whole `CI` workflow |
| `build_test` | `stable` plus `docs-prose.txt`, `logical-only.txt`, `not-linux-ci.txt` | `build_and_test` and `devdeps` |
| `docs` | `stable` plus `logical-only.txt`, `not-linux-ci.txt` | `documentation` and `devdeps` |
| macOS `scope` | `stable` plus `docs-prose.txt`, `logical-only.txt`, `not-macos-ci.txt` | all macOS jobs |

`CI Summary` removes the required checks of gated-off jobs from
`.github/required-checks.yml` before it verifies them, by name prefix. Keep
that list in step with the gates.

## Adding a gate

1. Find out what the job reads. Check the workflow, the scripts it calls and
   the CMake files, not only the directory names. For example,
   `docs/CMakeLists.txt` compiles the code under `docs/`, so only prose is
   skipped for the build.
2. List only paths that are safe to skip. A path that is missing from a list
   runs the jobs, a path that is wrongly listed hides a regression.
3. Add the gate to the `changes` job, re-export it from `metadata`, and put it
   in the `if:` of the job it controls.
