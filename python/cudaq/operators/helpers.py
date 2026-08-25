# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import inspect, numpy, os, re, sys, typing  # type: ignore
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

if (3, 10) <= sys.version_info:
    # type | type syntax is only supported in 3.10 or later
    NumericType = numpy.complexfloating | complex | float | int


def _aggregate_parameters(
        parameter_mappings: Iterable[Mapping[str, str]]) -> Mapping[str, str]:
    """
    Helper function used by all operator classes to return a mapping with the
    used parameters and their respective description as defined in a doc comment.
    """
    param_descriptions: dict[str, str] = {}
    for descriptions in parameter_mappings:
        for key, new_desc in descriptions.items():
            existing_desc = param_descriptions.get(key, "")
            if existing_desc and new_desc:
                param_descriptions[
                    key] = f'{existing_desc}{os.linesep}---{os.linesep}{new_desc}'
            else:
                param_descriptions[key] = new_desc or existing_desc
    return param_descriptions


def _parameter_docs(param_name: str, docs: Optional[str]) -> str:
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
        return r"(?:^\s*" + word + r":\s*\r?\n)"

    def param_pattern(param_name):
        return r"(?:^\s*" + param_name + r"\s*(\(.*\))?:)\s*(.*)$"

    param_docs = ""
    try:  # Make sure failing to retrieve docs never cases an error.
        split = re.split(keyword_pattern("Arguments|Args"),
                         docs,
                         flags=re.MULTILINE)
        if len(split) == 2:
            match = re.search(param_pattern(param_name), split[1], re.MULTILINE)
            if match is not None:
                param_docs = match.group(2) + split[1][match.end(2):]
                match = re.search(param_pattern("\\S*?"), param_docs,
                                  re.MULTILINE)
                if match is not None:
                    param_docs = param_docs[:match.start(0)]
                param_docs = re.sub(r'\s+', ' ', param_docs)
        return param_docs.strip()
    except Exception:
        return ""


def _args_from_kwargs(fct: Callable,
                      **kwargs: Any) -> Tuple[Sequence[Any], Mapping[str, Any]]:
    """
    Extracts the positional argument and keyword only arguments 
    for the given function from the passed `kwargs`. 
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
        # Try to get the argument from the `kwargs` passed during operator
        # evaluation.
        arg_value = kwargs.pop(arg_name, None)
        if arg_value is None:
            arg_value = signature.parameters[arg_name].default
            if arg_value is inspect.Parameter.empty:
                raise ValueError(f'missing keyword argument: {arg_name}')
        return arg_value

    extracted_args = [find_in_kwargs(arg_name) for arg_name in arg_spec.args]
    if consumes_kwargs:
        return extracted_args, kwargs
    elif len(arg_spec.kwonlyargs) > 0:
        # If we can't pass all remaining `kwargs`,
        # we need to create a separate dictionary for `kwonlyargs`.
        kwonlyargs = {
            arg_name: find_in_kwargs(arg_name)
            for arg_name in arg_spec.kwonlyargs
        }
        return extracted_args, kwonlyargs
    return extracted_args, {}
