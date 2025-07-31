# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import random
import math
import argparse
import re

gates = [
    {
        "name": "x",
        "ncontrols": 0,
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": False,
        "subcircuit": True,
        "weight": .2
    },
    {
        "name": "x",
        "ncontrols": 1,
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": False,
        "subcircuit": True,
        "weight": .2
    },
    {
        "name": "x",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "y",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "z",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "h",
        "ncontrols": 0,
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "s",
        "ncontrols": 0,
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": True,
        "subcircuit": False
    },
    {
        "name": "s",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": True,
        "subcircuit": False
    },
    {
        "name": "t",
        "ncontrols": 0,
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": True,
        "subcircuit": False
    },
    {
        "name": "t",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": True,
        "subcircuit": False
    },
    {
        "name": "rz",
        "ncontrols": 0,
        "ntargets": 1,
        "continuous": 1,
        "hasAdj": False,
        "subcircuit": True,
        "weight": .5
    },
    {
        "name": "rz",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 1,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "r1",
        "ncontrols": 0,
        "ntargets": 1,
        "continuous": 1,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "r1",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 1,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "rx",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 1,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "ry",
        "ncontrols": "any",
        "ntargets": 1,
        "continuous": 1,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "u3",
        "ncontrols": 0,
        "ntargets": 1,
        "continuous": 3,
        "hasAdj": False,
        "subcircuit": False
    },
    {
        "name": "swap",
        "ncontrols": 0,
        "ntargets": 2,
        "continuous": 0,
        "hasAdj": False,
        "subcircuit": True,
        "weight": .1
    },
]


def generate_indent(indent):
    return ''.join(['  ' for _ in range(indent)])


def generate_instruction(qubits, gates, subcircuit=False, indent=0):
    if subcircuit:
        gates = [gate for gate in gates if gate["subcircuit"]]
        weights = [gate["weight"] for gate in gates if gate["subcircuit"]]
        gate = random.choices(population=gates, k=1, weights=weights)[0]
    else:
        gate = random.choice(gates)
    gatestr = str(gate["name"])
    targets = random.sample(qubits, gate["ntargets"])
    ncontrols = gate["ncontrols"]
    control_choices = [qubit for qubit in qubits if qubit not in targets]
    if ncontrols == "any":
        if random.choice([True, False]):
            ncontrols = min(random.randint(0,
                                           len(control_choices) - 1),
                            random.randint(0,
                                           len(control_choices) - 1))
        else:
            ncontrols = 0
    controls = random.sample(control_choices, k=ncontrols)
    controlstrs = [str(control) for control in controls]
    continuous = [
        str(round(random.random() * 2 * math.pi, 2))
        for _ in range(gate["continuous"])
    ]
    operands = continuous + controlstrs + targets
    operandstr = ", ".join(operands)
    if ncontrols > 0:
        gatestr += "<cudaq::ctrl>"
    elif gate["hasAdj"]:
        if random.choice([True, False]):
            gatestr += "<cudaq::adj>"

    return generate_indent(indent) + gatestr + "(" + operandstr + ");"


def generate_qubit_choices(oldQubits, newQubits):
    return ["q[{}]".format(str(n + oldQubits)) for n in range(newQubits)]


def generate_subcircuit(qubits, nGates, indent=0):
    block = generate_block(qubits, nGates, True, indent=indent)
    breakers = [
        generate_indent(indent) + "h({});".format(qubit) for qubit in qubits
    ]
    return block + "\n".join(breakers) + "\n"


def generate_block(qubits, nGates, subcircuit=False, indent=0):
    return "\n".join([
        generate_instruction(qubits, gates, subcircuit, indent=indent)
        for _ in range(nGates)
    ]) + "\n"


def generate_inst_list(qubits, nGates, nBlocks):
    program = ""
    for _ in range(nBlocks):
        if (random.choice([True, False])):
            program += generate_subcircuit(qubits, nGates) + "\n"
        else:
            program += generate_block(qubits, nGates) + "\n"
    return program


def generate_qvector(nQubits, indent=0):
    return generate_indent(indent) + "cudaq::qvector q({});\n".format(
        str(nQubits))


def generate_measures(qubits, indent=0):
    measures = [
        generate_indent(indent) + "mz({});".format(qubit) for qubit in qubits
    ]
    return "\n".join(measures) + "\n"


def parse_range(numbers: str):
    if '-' in numbers:
        xr = numbers.split('-')
        return range(int(xr[0].strip()), int(xr[1].strip()) + 1)
    res = []
    for x in numbers.split(','):
        x = x.strip()
        if x.isdigit():
            res.append(int(x))
        else:
            raise ValueError(f"Unknown range specified: {x}")
    return res


argparser = argparse.ArgumentParser(prog='RandomCircuitGenerator',
                                    description='Generates random circuits',
                                    epilog='')

argparser.add_argument('template', type=str)
argparser.add_argument('--seed', type=int)
args = argparser.parse_args()

program = ""
with open(args.template, 'r') as file:
    random.seed(args.seed)
    indent = 0
    working_qubits = []
    qubit_stack = []
    for raw in file:
        line = raw.strip()

        if '}' in line:
            indent -= 1
            working_qubits = qubit_stack.pop()

        if line.startswith("GEN-QALLOC: "):
            #res = parse("GEN-QALLOC: nqubits={qubits}", line)
            match = re.search(r"GEN-QALLOC: nqubits=(\d+)", line)
            if match:
                nQubits = int(match.group(1))
            else:
                raise RuntimeError(
                    "Illegal 'GEN-QALLOC' command, expected 'GEN-QALLOC: nqubits=..., got '{}'"
                    .format(line))
            #nQubits = random.choice(parse_range(res['qubits']))
            qubits = generate_qubit_choices(len(working_qubits), nQubits)
            working_qubits = working_qubits + qubits
            program += generate_qvector(nQubits, indent)
        elif line.startswith("GEN-BLOCK: "):
            match = re.search(r"GEN-BLOCK: <(.*)>", line)
            if match:
                nGates = match.group(1)
            else:
                raise RuntimeError(
                    "Illegal 'GEN-BLOCK' command, expected 'GEN-BLOCK: <range>, got '{}'"
                    .format(line))
            nGates = random.choice(parse_range(nGates))
            program += generate_block(working_qubits, nGates, indent=indent)
        elif line.startswith("GEN-SUBCIRCUIT: "):
            match = re.search(r"GEN-SUBCIRCUIT: <(.*)>", line)
            if match:
                nGates = match.group(1)
            else:
                raise RuntimeError(
                    "Illegal 'GEN-SUBCIRCUIT' command, expected 'GEN-SUBCIRCUIT: <range>, got '{}'"
                    .format(line))
            nGates = random.choice(parse_range(nGates))
            program += generate_subcircuit(working_qubits,
                                           nGates,
                                           indent=indent)
        elif line.startswith("GEN-MEASURES"):
            program += generate_measures(working_qubits, indent=indent)
        else:
            if len(working_qubits) > 0:
                line = line.replace("GEN:<nqubits>", str(len(working_qubits)))
                line = line.replace("GEN:<qubit>",
                                    random.choice(working_qubits))
            program += generate_indent(indent) + line + "\n"

        if '{' in line:
            indent += 1
            qubit_stack.append(working_qubits)

print(program)
