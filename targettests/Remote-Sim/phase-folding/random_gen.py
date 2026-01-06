# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
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
    },
    {
        "name": "x",
        "ncontrols": 1,
        "ntargets": 1,
        "continuous": 0,
        "hasAdj": False,
        "subcircuit": True,
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
    },
]


def generate_indent(indent):
    return ''.join(['  ' for _ in range(indent)])


def generate_instruction(qubits,
                         gates,
                         subcircuit=False,
                         rz_weight=None,
                         indent=0):
    if subcircuit:
        if random.random() <= rz_weight:
            gate = {
                "name": "rz",
                "ncontrols": 0,
                "ntargets": 1,
                "continuous": 1,
                "hasAdj": False,
                "subcircuit": True,
            }
        else:
            sc_gates = [
                gate for gate in gates
                if gate["subcircuit"] and gate["name"] != "rz"
            ]
            gate = random.choices(population=sc_gates, k=1)[0]
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


def generate_qubit_choices(name, newQubits):
    return ["{}[{}]".format(name, str(n)) for n in range(newQubits)]


idx = 0


def generate_qvector(nQubits, indent=0):
    global idx
    qvector = "q" + str(idx)
    idx += 1
    decl = generate_indent(indent) + "cudaq::qvector {}({});\n".format(
        qvector, str(nQubits))
    choices = generate_qubit_choices(qvector, nQubits)
    return [qvector, decl, choices]


def generate_subcircuit(qubits, nGates, rz_weight, indent=0):
    return generate_block(qubits,
                          nGates,
                          True,
                          rz_weight=rz_weight,
                          indent=indent)
    # breakers = [
    #     generate_indent(indent) + "h({});".format(qubit) for qubit in qubits
    # ]
    # return block + "\n".join(breakers) + "\n"


def generate_block(qubits, nGates, subcircuit=False, rz_weight=None, indent=0):
    return "\n".join([
        generate_instruction(
            qubits, gates, subcircuit, rz_weight=rz_weight, indent=indent)
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
argparser.add_argument('--block-length', type=str)
argparser.add_argument('--rz-weight', type=float)
args = argparser.parse_args()

block_length = parse_range(args.block_length)
rz_weight = args.rz_weight
program = ""
with open(args.template, 'r') as file:
    random.seed(args.seed)
    indent = 0
    working_qubits = []
    qubit_stack = []
    qvectors = []
    qvector_stack = []
    for raw in file:
        line = raw.strip()

        if '}' in line:
            indent -= 1
            working_qubits = qubit_stack.pop()
            qvectors = qvector_stack.pop()

        if line.startswith("GEN-QALLOC: "):
            match = re.search(r"GEN-QALLOC: nqubits=(\d+)", line)
            if match:
                nQubits = int(match.group(1))
            else:
                raise RuntimeError(
                    "Illegal 'GEN-QALLOC' command, expected 'GEN-QALLOC: nqubits=..., got '{}'"
                    .format(line))
            [qvector, decl, qubits] = generate_qvector(nQubits, indent)
            working_qubits = working_qubits + qubits
            qvectors.append(qvector)
            program += decl
        elif line.startswith("GEN-BLOCK"):
            nGates = random.choice(block_length)
            program += generate_block(working_qubits, nGates, indent=indent)
        elif line.startswith("GEN-SUBCIRCUIT"):
            nGates = random.choice(block_length)
            program += generate_subcircuit(working_qubits,
                                           nGates,
                                           rz_weight,
                                           indent=indent)
        elif line.startswith("GEN-MEASURES"):
            program += generate_measures(working_qubits, indent=indent)
        else:
            if len(working_qubits) > 0:
                line = line.replace("GEN:<nqubits>", str(len(working_qubits)))
                line = line.replace("GEN:<qubit>",
                                    random.choice(working_qubits))
            if len(qvectors) > 0:
                line = line.replace("GEN:<qvector>", random.choice(qvectors))
            program += generate_indent(indent) + line + "\n"

        if '{' in line:
            indent += 1
            qubit_stack.append(working_qubits)
            qvector_stack.append(qvectors)

print(program)
