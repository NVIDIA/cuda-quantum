# Available compiler passes

The CUDA-Q compiler pass catalog is generated from the Transform and
code-generation `Passes.td` definitions during the documentation build. It
records each built-in pass's textual argument, description, and options.
Operation anchors and other implementation details remain in the TableGen
definitions. The catalog does not assign passes to target stages because target
pipelines compose passes according to their own IR requirements.

See {doc}`Developing compiler passes <mlir_pass>` for the workflow used to
design, implement, test, and integrate a pass.

## Built-in passes

```{contents} Pass index
:local:
:depth: 1
```

```{include} /_mdgen/Transforms.md
```

```{include} /_mdgen/CodeGenPasses.md
```
