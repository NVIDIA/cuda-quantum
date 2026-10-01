# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q kernel declarations used as CUDA-Q Logical objectives."""

from __future__ import annotations

from inspect import Signature, signature
from typing import get_args, get_origin

from .definition import ProgramDefinition


def kernel_declaration_from_kernel(kernel):
    """Return a body-less CUDA-Q declaration without mutating ``kernel``."""

    from cudaq.kernel.kernel_decorator import (
        PyKernelDecorator,
        isa_kernel_decorator,
    )
    from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
    from cudaq.mlir.ir import InsertionPoint, Operation, StringAttr, UnitAttr

    if not isa_kernel_decorator(kernel):
        raise TypeError("expected an @cudaq.kernel")
    declaration = PyKernelDecorator(
        None,
        module=cudaq_runtime.cloneModule(kernel.qkeModule),
        kernelName=kernel.name,
        decorator=kernel,
    )
    functions = [
        view for view in declaration.qkeModule.body.operations
        if view.operation.name == "func.func" and
        "cudaq-kernel" in view.operation.attributes
    ]
    if len(functions) != 1:
        raise ValueError(
            "a CUDA-Q objective kernel must contain exactly one kernel function"
        )
    function = functions[0]
    attributes = dict(function.operation.attributes)
    attributes["sym_visibility"] = StringAttr.get(
        "private", context=declaration.qkeModule.context)
    attributes["qlx-objective"] = UnitAttr.get(
        context=declaration.qkeModule.context)
    with declaration.qkeModule.context, InsertionPoint(function.operation):
        Operation.create("func.func",
                         attributes=attributes,
                         regions=1,
                         loc=function.operation.location)
    function.operation.erase()
    return declaration


def _flatten_logical_results(annotation):
    from ..types.values import logical_qubit

    if annotation in (Signature.empty, None):
        return ()
    if annotation is logical_qubit:
        return (annotation,)
    if get_origin(annotation) is tuple:
        values = get_args(annotation)
        if len(values) == 2 and values[1] is Ellipsis:
            raise TypeError(
                "kernel declarations require fixed-size objective results")
        result = []
        for value in values:
            result.extend(_flatten_logical_results(value))
        return tuple(result)
    return (annotation,)


def kernel_declaration_from_objective(definition: ProgramDefinition):
    """Return a CUDA-Q declaration for one action-like P0 objective."""

    from cudaq.kernel.kernel_decorator import PyKernelDecorator

    from ..types.values import logical_qubit

    if (definition.kind != "objective" or
            definition.objective_kind == "instrument"):
        return None
    if definition.cudaq_kernel is not None:
        return kernel_declaration_from_kernel(definition.cudaq_kernel)
    if definition.objective_kind == "auto":
        from ..compiler import compile

        if compile(definition).root.kind != "action":
            return None
    if not definition.name.isidentifier():
        raise ValueError(
            "CUDA-Q-callable objective names must be Python identifiers")
    parameters = tuple(definition.signature.parameters.values())
    if (not parameters or any(
            definition.type_hints.get(parameter.name, parameter.annotation)
            is not logical_qubit for parameter in parameters)):
        return None
    results = _flatten_logical_results(
        definition.type_hints.get("return",
                                  definition.signature.return_annotation))
    if len(results) != len(parameters) or any(
            result is not logical_qubit for result in results):
        return None
    arguments = ", ".join(
        f"{parameter.name}: cudaq.qubit" for parameter in parameters)
    source = f"def {definition.name}({arguments}):\n    pass\n"
    stub = PyKernelDecorator(source, kernelName=definition.name)
    return kernel_declaration_from_kernel(stub)


def _logical_signature(kernel):
    import cudaq

    from ..types.values import logical_qubit

    provider = getattr(kernel, "kernelFunction", None)
    if provider is None:
        raise TypeError(
            "gadget implements= requires a Python-authored @cudaq.kernel")
    source_signature = signature(provider)
    annotation_namespace = dict(provider.__globals__)
    annotation_namespace.update(
        {alias: cudaq for alias in getattr(kernel, "cudaqAliases", ("cudaq",))})
    annotation_namespace.setdefault("qubit", cudaq.qubit)

    def resolve(annotation):
        if not isinstance(annotation, str):
            return annotation
        try:
            return eval(annotation, annotation_namespace)
        except (NameError, TypeError, SyntaxError):
            return annotation

    parameters = []
    hints = {}
    quantum_inputs = []
    for parameter in source_signature.parameters.values():
        annotation = resolve(parameter.annotation)
        if annotation is cudaq.qubit:
            annotation = logical_qubit
            quantum_inputs.append(parameter.name)
        elif annotation is Signature.empty:
            raise TypeError(
                f"CUDA-Q objective kernel parameter {parameter.name!r} requires "
                "a type annotation")
        else:
            raise TypeError(
                "CUDA-Q objective kernels currently accept only cudaq.qubit "
                f"operands; {parameter.name!r} has annotation {annotation!r}")
        hints[parameter.name] = annotation
        parameters.append(parameter.replace(annotation=annotation))
    if not quantum_inputs:
        raise TypeError(
            "CUDA-Q objective kernels currently require at least one cudaq.qubit "
            "operand")
    return_annotation = resolve(source_signature.return_annotation)
    if return_annotation not in (Signature.empty, None):
        raise TypeError(
            "CUDA-Q objective kernels currently use in-place qubit results and "
            "cannot declare a Python return value")
    result = (logical_qubit if len(quantum_inputs) == 1 else tuple[tuple(
        logical_qubit for _ in quantum_inputs)])
    hints["return"] = result
    return source_signature.replace(parameters=parameters,
                                    return_annotation=result), hints


def program_definition_from_kernel(kernel) -> ProgramDefinition:
    """Create the lazy P0 objective for a CUDA-Q implementation kernel."""

    from cudaq.kernel.kernel_decorator import isa_kernel_decorator

    if not isa_kernel_decorator(kernel):
        raise TypeError("gadget implements= expects a QLX objective or "
                        "@cudaq.kernel")
    logical_signature, hints = _logical_signature(kernel)

    def provider(*_args, **_kwargs):
        raise RuntimeError("CUDA-Q objective kernels materialize from Quake")

    provider.__name__ = kernel.name
    provider.__qualname__ = kernel.name
    provider.__doc__ = getattr(kernel.kernelFunction, "__doc__", None)
    provider.__module__ = getattr(kernel.kernelFunction, "__module__", __name__)
    provider.__signature__ = logical_signature
    provider.__annotations__ = dict(hints)
    definition = ProgramDefinition(provider,
                                   kind="objective",
                                   objective_kind="auto",
                                   name=kernel.name,
                                   type_hints=hints,
                                   cudaq_kernel=kernel)
    return definition


def _objective_wrapper(kernel, arity: int):
    from cudaq.kernel.kernel_decorator import PyKernelDecorator

    allocations = "".join(
        f"    qubit_{index} = cudaq.qubit()\n" for index in range(arity))
    arguments = ", ".join(f"qubit_{index}" for index in range(arity))
    wrapper_name = f"__qlx_{kernel.name}_objective"
    source = (f"def {wrapper_name}():\n"
              f"{allocations}"
              f"    kernel({arguments})\n")
    return PyKernelDecorator(source, kernelName=wrapper_name)


def materialize_kernel_objective(definition: ProgramDefinition):
    """Lower a retained CUDA-Q kernel body and expose it as a P0 action."""

    from cudaq.mlir import ir as mlir_ir
    from ..algebra.clifford import CliffordAction, NonCliffordAction
    from ..compiler.build import Build, EvidenceRecord
    from ..compiler.context import CompilationContext
    from ..compiler.pipeline import pipelines
    from ..compiler.quake import import_cudaq
    from cudaq.mlir.dialects.qlx import ObjectiveBodyOp, ReturnOp
    from .definition import DefinitionHandle

    kernel = definition.cudaq_kernel
    if kernel is None:
        raise TypeError("definition is not backed by a CUDA-Q objective kernel")
    arity = len(definition.signature.parameters)
    imported = import_cudaq(_objective_wrapper(kernel, arity), _objective=True)
    module = imported._fresh_module()
    transaction = CompilationContext(module=module)
    program = transaction.find_symbol(imported.root.symbol, "qlx.program")
    if program is None:
        raise ValueError("CUDA-Q objective import produced no P0 program")
    block = program.regions[0].blocks[0]
    operations = tuple(view.operation for view in block.operations)
    prepares = [
        operation for operation in operations if operation.name == "qlx.prepare"
    ]
    discards = [
        operation for operation in operations if operation.name == "qlx.discard"
    ]
    if len(prepares) != arity or len(discards) != 1:
        raise TypeError(
            "CUDA-Q objective kernel must preserve exactly its input qubits and "
            "must not allocate, measure, or discard additional qubits")

    logical_type = mlir_ir.Type.parse("!qlx.logical_qubit",
                                      context=module.context)
    inputs = tuple(logical_type for _ in range(arity))
    results = inputs
    function_type = mlir_ir.FunctionType.get(inputs,
                                             results,
                                             context=module.context)
    body_symbol = transaction.unique_symbol(f"{definition.name}_objective_body")
    with module.context, mlir_ir.InsertionPoint(program):
        objective = ObjectiveBodyOp(
            body_symbol,
            mlir_ir.TypeAttr.get(function_type),
            objective_kind="action",
            loc=program.location,
        )
    objective.operation.attributes["qlx.stage"] = mlir_ir.StringAttr.get(
        "p0", context=module.context)
    objective.operation.attributes["qlx.profile"] = mlir_ir.StringAttr.get(
        "p0", context=module.context)
    block.append_to(objective.body)

    arguments = tuple(
        block.add_argument(logical_type,
                           mlir_ir.Location.unknown(context=module.context))
        for _ in range(arity))
    ordered_prepares = sorted(
        prepares, key=lambda operation: int(operation.attributes["allocation"]))

    def terminal_owner(value):
        """Follow one in-place kernel argument to its final linear owner."""

        visited = set()
        while value not in visited:
            visited.add(value)
            uses = tuple(value.uses)
            if len(uses) != 1:
                raise TypeError(
                    "CUDA-Q objective kernel arguments must have one linear "
                    "successor at every operation")
            use = uses[0]
            owner = getattr(use.owner, "operation", use.owner)
            if owner.name == "qlx.discard":
                return value
            if owner.name != "qlx.apply" or use.operand_number >= len(
                    owner.results):
                raise TypeError(
                    "CUDA-Q objective kernel arguments may flow only through "
                    "in-place logical actions")
            value = owner.results[use.operand_number]
        raise TypeError("CUDA-Q objective kernel contains a cyclic owner flow")

    # CUDA-Q is free to order the compiler-generated sinks independently of
    # the source arguments. Recover each argument's final owner from its linear
    # SSA chain before removing the wrapper allocation/discard boundary.
    outputs = tuple(
        terminal_owner(prepare.results[0]) for prepare in ordered_prepares)
    for prepare, argument in zip(ordered_prepares, arguments):
        prepare.results[0].replace_all_uses_with(argument)
        prepare.erase()

    if len(outputs) != arity:
        raise TypeError(
            "CUDA-Q objective kernel did not return every input qubit owner")
    discards[0].erase()
    terminator = block.operations[-1].operation
    if terminator.name != "qlx.return":
        raise ValueError("CUDA-Q objective import has no P0 return terminator")
    with module.context, mlir_ir.InsertionPoint(terminator):
        ReturnOp(outputs, loc=terminator.location)
    terminator.erase()
    program.erase()
    transaction._index_symbol(objective.operation, body_symbol)

    action_symbol = transaction.declare_objective(
        family="action",
        requested_symbol=definition.name,
        kind="composite",
        inputs=inputs,
        results=results,
        semantics=mlir_ir.FlatSymbolRefAttr.get(body_symbol,
                                                context=module.context),
    )
    try:
        clifford = CliffordAction.from_mlir_program(
            objective.operation,
            ports=tuple(definition.signature.parameters),
        )
    except NonCliffordAction:
        pass
    else:
        action = transaction.find_symbol(action_symbol, "qlx.action")
        action.attributes["clifford_action"] = clifford.to_mlir_attr(
            module.context)

    return Build(
        context=module.context,
        module=module,
        root=DefinitionHandle(action_symbol, "action", "p0"),
        profile="p0",
        pipeline=pipelines.logical(),
        evidence=(*imported.evidence,
                  EvidenceRecord(
                      kind="kernel_objective_derivation",
                      producer="qlx-python@0.3",
                      result="pass",
                      obligations=("p0-objective", "kernel-call-boundary"),
                  )),
        source_modules=imported.source_modules,
    )


__all__ = []
