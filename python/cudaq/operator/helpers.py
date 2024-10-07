# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import inspect, itertools, numpy, os, re, sys  # type: ignore
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from numpy.typing import NDArray

from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

if (3, 11) <= sys.version_info:
    NumericType = SupportsComplex
else:
    NumericType = numpy.complexfloating | complex | float | int


class _OperatorHelpers:

    @staticmethod
    def aggregate_parameters(
            parameter_mappings: Iterable[Mapping[str,
                                                 str]]) -> Mapping[str, str]:
        """
        Helper function used by all operator classes to return a mapping with the
        used parameters and their respective description as defined in a doc comment.
        """
        param_descriptions: dict[str, str] = {}
        for descriptions in parameter_mappings:
            for key in descriptions:
                existing_desc = param_descriptions.get(key) or ""
                new_desc = descriptions[key]
                has_existing_desc = existing_desc is not None and existing_desc != ""
                has_new_desc = new_desc != ""
                if has_existing_desc and has_new_desc:
                    param_descriptions[
                        key] = existing_desc + f'{os.linesep}---{os.linesep}' + new_desc
                elif has_existing_desc:
                    param_descriptions[key] = existing_desc
                else:
                    param_descriptions[key] = new_desc
        return param_descriptions

    @staticmethod
    def parameter_docs(param_name: str, docs: Optional[str]) -> str:
        """
        Given the function documentation, tries to find the documentation for a 
        specific parameter. Expects Google docs style to be used.

        Returns:
            The documentation or an empty string if it failed to find it.
        """
        if param_name is None or docs is None:
            return ""

        # Using re.compile on the pattern before passing it to search
        # seems to behave slightly differently than passing the string.
        # Also, Python caches already used patterns, so compiling on
        # the fly seems fine.
        def keyword_pattern(word):
            return r"(?:^\\s*" + word + ":\\s*\r?\n)"

        def param_pattern(param_name):
            return r"(?:^\s*" + param_name + r"\s*(\(.*\))?:)\s*(.*)$"

        param_docs = ""
        try:  # Make sure failing to retrieve docs never cases an error.
            split = re.split(keyword_pattern("Arguments"),
                             docs,
                             flags=re.MULTILINE)
            if len(split) == 1:
                split = re.split(keyword_pattern("Args"),
                                 docs,
                                 flags=re.MULTILINE)
            if len(split) > 1:
                match = re.search(param_pattern(param_name), split[1],
                                  re.MULTILINE)
                if match is not None:
                    param_docs = match.group(2) + split[1][match.end(2):]
                    match = re.search(param_pattern("\\S*?"), param_docs,
                                      re.MULTILINE)
                    if match is not None:
                        param_docs = param_docs[:match.start(0)]
                    param_docs = re.sub(r'\s+', ' ', param_docs)
        except:
            pass
        return param_docs.strip()

    @staticmethod
    def args_from_kwargs(
            fct: Callable,
            **kwargs: Any) -> tuple[Sequence[Any], Mapping[str, Any]]:
        """
        Extracts the positional argument and keyword only arguments 
        for the given function from the passed kwargs. 
        """
        arg_spec = inspect.getfullargspec(fct)
        signature = inspect.signature(fct)
        if arg_spec.varargs is not None:
            raise ValueError(
                "cannot extract arguments for a function with a *args argument")
        consumes_kwargs = arg_spec.varkw is not None
        if consumes_kwargs:
            kwargs = kwargs.copy()  # We will modify and return a copy

        def find_in_kwargs(arg_name: str) -> Any:
            # Try to get the argument from the kwargs passed during operator
            # evaluation.
            arg_value = kwargs.get(arg_name)
            if arg_value is None:
                # If no suitable keyword argument was defined, check if the
                # generator defines a default value for this argument.
                default_value = signature.parameters[arg_name].default
                if default_value is not inspect.Parameter.empty:
                    arg_value = default_value
            elif consumes_kwargs:
                del kwargs[arg_name]
            if arg_value is None:
                raise ValueError(f'missing keyword argument {arg_name}')
            return arg_value

        extracted_args = []
        for arg_name in arg_spec.args:
            extracted_args.append(find_in_kwargs(arg_name))
        if consumes_kwargs:
            return extracted_args, kwargs
        elif len(arg_spec.kwonlyargs) > 0:
            # If we can't pass all remaining kwargs,
            # we need to create a separate dictionary for kwonlyargs.
            kwonlyargs: dict[str, Any] = {}
            for arg_name in arg_spec.kwonlyargs:
                kwonlyargs[arg_name] = find_in_kwargs(arg_name)
            return extracted_args, kwonlyargs
        return extracted_args, {}

    @staticmethod
    def generate_all_states(degrees: Sequence[int],
                            dimensions: Mapping[int, int]) -> tuple[str]:
        """
        Generates all possible states for the given dimensions ordered according to 
        the sequence of degrees (ordering is relevant if dimensions differ).
        """
        if len(degrees) == 0:
            return []
        states = [[str(state)] for state in range(dimensions[degrees[0]])]
        for d in degrees[1:]:
            prod = itertools.product(
                states, [str(state) for state in range(dimensions[d])])
            states = [current + [new] for current, new in prod]
        return tuple((''.join(state) for state in states))

    @staticmethod
    def permute_matrix(matrix: NDArray[numpy.complexfloating],
                       permutation: Iterable[int]) -> None:
        """
        Permutes the given matrix according to the given permutation.
        If states is the current order of vector entries on which the given matrix
        acts, and permuted_states is the desired order of an array on which the
        permuted matrix should act, then the permutation is defined such that
        [states[i] for i in permutation] produces permuted_states.
        """
        for i in range(numpy.size(matrix, 1)):
            matrix[:, i] = matrix[permutation, i]
        for i in range(numpy.size(matrix, 0)):
            matrix[i, :] = matrix[i, permutation]

    @staticmethod
    def cmatrix_to_nparray(
            cmatrix: cudaq_runtime.ComplexMatrix
    ) -> NDArray[numpy.complexfloating]:
        """
        Converts a `cudaq.ComplexMatrix` to the corresponding numpy array.
        """
        # FIXME: implement conversion in py_matrix.cpp instead and ensure consistency with numpy.array -> ComplexMatrix
        return numpy.array(
            [[cmatrix[row, column]
              for row in range(cmatrix.num_rows())]
             for column in range(cmatrix.num_columns())],
            dtype=numpy.complex128)

    @staticmethod
    def canonicalize_degrees(degrees: Iterable[int]) -> tuple[int]:
        """
        Returns the degrees sorted in canonical order.
        """
        return tuple(sorted(degrees, reverse=True))
