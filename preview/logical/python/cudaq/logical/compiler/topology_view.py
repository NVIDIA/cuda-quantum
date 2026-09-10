# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed, read-only graph views over canonical Fabric and Phys IR facts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping


def _text(attribute) -> str:
    value = getattr(attribute, "value", None)
    text = str(attribute if value is None else value)
    return text.strip('"').lstrip("@").split("::@")[-1]


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for view in block.operations:
                yield from _walk(view.operation)


def _array(attribute) -> tuple:
    return tuple(attribute) if attribute is not None else ()


def _integer(attribute) -> int:
    value = getattr(attribute, "value", attribute)
    return int(value)


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    kind: str
    attributes: Mapping[str, Any]

    def __init__(self, id: str, kind: str, attributes: Mapping[str, Any] = ()):
        object.__setattr__(self, "id", str(id))
        object.__setattr__(self, "kind", str(kind))
        object.__setattr__(self, "attributes",
                           MappingProxyType(dict(attributes)))


@dataclass(frozen=True, slots=True)
class GraphEdge:
    id: str
    kind: str
    endpoints: tuple[str, ...]
    attributes: Mapping[str, Any]

    def __init__(
            self,
            id: str,
            kind: str,
            endpoints,
            attributes: Mapping[str, Any] = (),
    ):
        endpoints = tuple(map(str, endpoints))
        if len(endpoints) < 2:
            raise ValueError("graph edges require at least two endpoints")
        object.__setattr__(self, "id", str(id))
        object.__setattr__(self, "kind", str(kind))
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "attributes",
                           MappingProxyType(dict(attributes)))


class _GraphView:
    __slots__ = ("nodes", "edges", "metadata")

    def __init__(self, nodes, edges, *, metadata=()):
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.metadata = MappingProxyType(dict(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata":
                dict(self.metadata),
            "nodes": [{
                "id": node.id,
                "kind": node.kind,
                **dict(node.attributes)
            } for node in self.nodes],
            "edges": [{
                "id": edge.id,
                "kind": edge.kind,
                "endpoints": list(edge.endpoints),
                **dict(edge.attributes),
            } for edge in self.edges],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_dot(self) -> str:

        def quoted(value):
            return json.dumps(str(value))

        lines = ["graph qlx {"]
        for node in self.nodes:
            label = f"{node.id}\\n{node.kind}"
            lines.append(f"  {quoted(node.id)} [label={quoted(label)}];")
        for edge in self.edges:
            anchor = edge.endpoints[0]
            for ordinal, endpoint in enumerate(edge.endpoints[1:]):
                label = edge.kind if ordinal == 0 else f"{edge.kind} ({ordinal + 1})"
                lines.append(f"  {quoted(anchor)} -- {quoted(endpoint)} "
                             f"[label={quoted(label)}, id={quoted(edge.id)}];")
        lines.append("}")
        return "\n".join(lines)

    def to_networkx(self):
        """Return a NetworkX MultiGraph when the optional package is present."""

        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "to_networkx() requires the optional 'networkx' package"
            ) from exc
        graph = nx.MultiGraph(**dict(self.metadata))
        for node in self.nodes:
            graph.add_node(node.id, kind=node.kind, **dict(node.attributes))
        for edge in self.edges:
            anchor = edge.endpoints[0]
            for ordinal, endpoint in enumerate(edge.endpoints[1:]):
                graph.add_edge(
                    anchor,
                    endpoint,
                    key=f"{edge.id}:{ordinal}",
                    id=edge.id,
                    kind=edge.kind,
                    hyperedge_endpoints=edge.endpoints,
                    **dict(edge.attributes),
                )
        return graph

    def write_png(
        self,
        path,
        *,
        positions: Mapping[str, tuple[float, float]] | None = None,
        title: str | None = None,
        dpi: int = 160,
    ):
        """Render this typed view to a PNG using optional visualization packages.

        ``positions`` maps graph node identities to ``(x, y)`` coordinates.  A
        deterministic spring layout is used when positions are omitted.  The
        renderer colors patch ownership by implicit slot and distinguishes the
        authored carrier-architecture layer from derived physical events.
        """

        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.patches import FancyArrowPatch, Patch as LegendPatch
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "write_png() requires the optional 'visualization' dependencies"
            ) from exc

        identities = {node.id for node in self.nodes}
        if positions is None:
            layout = nx.spring_layout(self.to_networkx(), seed=7)
            positions = {
                identity: (float(point[0]), float(point[1]))
                for identity, point in layout.items()
            }
        else:
            positions = {
                str(identity): (float(point[0]), float(point[1]))
                for identity, point in positions.items()
            }
            missing = identities - positions.keys()
            if missing:
                raise ValueError(
                    "positions must cover every graph node; missing " +
                    ", ".join(sorted(missing)))

        slot_palette = (
            "#76B900",
            "#4C78A8",
            "#F58518",
            "#B279A2",
            "#54A24B",
            "#E45756",
        )

        def slot_of(node):
            return node.attributes.get("slot",
                                       node.attributes.get("patch_slot"))

        slots = sorted({
            int(slot)
            for node in self.nodes
            if (slot := slot_of(node)) is not None
        })
        slot_colors = {
            slot: slot_palette[ordinal % len(slot_palette)]
            for ordinal, slot in enumerate(slots)
        }
        layer_colors = {
            "p1": "#4C78A8",
            "p2": "#76B900",
            "p3": "#F58518",
        }

        width = max(7.0, min(13.0, 1.0 + 0.85 * len(self.nodes)))
        height = max(4.2, min(8.5, 2.8 + 0.36 * len(self.nodes)))
        figure, axis = plt.subplots(figsize=(width, height))
        axis.set_facecolor("#FAFAFA")

        event_counts = {}
        for edge in self.edges:
            anchor = edge.endpoints[0]
            for endpoint in edge.endpoints[1:]:
                if edge.attributes.get("layer") == "event":
                    key = (anchor, endpoint, edge.kind)
                    event_counts[key] = event_counts.get(key, 0) + 1
                    continue
                start = positions[anchor]
                finish = positions[endpoint]
                directed = bool(edge.attributes.get("directed", False))
                architecture = edge.attributes.get("layer") == "architecture"
                color = "#6F7782" if architecture else "#2F5D50"
                line = FancyArrowPatch(
                    start,
                    finish,
                    arrowstyle="-|>" if directed else "-",
                    mutation_scale=14,
                    color=color,
                    linewidth=2.2 if architecture else 2.8,
                    shrinkA=19,
                    shrinkB=19,
                    zorder=1,
                )
                axis.add_patch(line)
                actions = tuple(edge.attributes.get("actions", ()))
                if actions:
                    label = "/".join(str(action).upper() for action in actions)
                else:
                    pair_count = len(edge.attributes.get("carrier_pairs", ()))
                    label = edge.kind.upper()
                    if pair_count:
                        pair_kind = ("role pairs" if self.metadata.get("stage")
                                     == "p2" else "carrier pairs")
                        label += f" · {pair_count} {pair_kind}"
                midpoint = (
                    (start[0] + finish[0]) / 2,
                    (start[1] + finish[1]) / 2 - 0.10,
                )
                axis.text(
                    *midpoint,
                    label,
                    fontsize=8,
                    color=color,
                    ha="center",
                    va="top",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.88,
                    },
                    zorder=4,
                )

        event_palette = {"swap": "#F58518", "cx": "#7A3E9D"}
        for (anchor, endpoint, kind), count in sorted(event_counts.items()):
            start = positions[anchor]
            finish = positions[endpoint]
            color = event_palette.get(kind.lower(), "#D62728")
            line = FancyArrowPatch(
                start,
                finish,
                arrowstyle="-",
                connectionstyle="arc3,rad=0.24",
                color=color,
                linewidth=2.5,
                linestyle="--",
                shrinkA=19,
                shrinkB=19,
                zorder=2,
            )
            axis.add_patch(line)
            midpoint = (
                (start[0] + finish[0]) / 2,
                (start[1] + finish[1]) / 2 + 0.14,
            )
            label = kind.upper() + (f" ×{count}" if count > 1 else "")
            axis.text(
                *midpoint,
                label,
                fontsize=8,
                color=color,
                weight="bold",
                ha="center",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                zorder=4,
            )

        for node in self.nodes:
            x, y = positions[node.id]
            slot = slot_of(node)
            color = (slot_colors[int(slot)] if slot is not None else
                     layer_colors.get(node.attributes.get("layer"), "#D8DCE2"))
            size = 1500 if node.kind == "carrier" else 2700
            axis.scatter(
                [x],
                [y],
                s=size,
                color=color,
                edgecolor="#263238",
                linewidth=1.2,
                zorder=3,
            )
            if node.kind == "carrier":
                label = f"q{node.attributes.get('index', node.id)}"
                label += f"\nslot {slot}" if slot is not None else "\nroute"
                roles = tuple(node.attributes.get("roles", ()))
                if roles:
                    label += "\n" + str(roles[0]).split(".", 1)[-1]
            else:
                label = str(node.attributes.get("label", node.id))
                if slot is not None:
                    label += f"\nPatch · slot {slot}"
                code = node.attributes.get("code")
                if code:
                    label += f"\n{code}"
            axis.text(
                x,
                y,
                label,
                fontsize=8.5,
                ha="center",
                va="center",
                color="#111820",
                zorder=4,
            )

        legend = [
            LegendPatch(
                facecolor=slot_colors[slot],
                edgecolor="#263238",
                label=f"patch slot {slot}",
            ) for slot in slots
        ]
        if any(slot_of(node) is None for node in self.nodes):
            present_layers = tuple(layer for layer in ("p1", "p2", "p3") if any(
                node.attributes.get("layer") == layer and slot_of(node) is None
                for node in self.nodes))
            legend.extend(
                LegendPatch(
                    facecolor=layer_colors[layer],
                    edgecolor="#263238",
                    label=f"{layer.upper()} layer",
                ) for layer in present_layers)
            if any(
                    slot_of(node) is None and
                    node.attributes.get("layer") not in layer_colors
                    for node in self.nodes):
                legend.append(
                    LegendPatch(
                        facecolor="#D8DCE2",
                        edgecolor="#263238",
                        label="unassigned carrier",
                    ))
        if any(
                edge.attributes.get("layer") == "architecture"
                for edge in self.edges):
            legend.append(
                Line2D([0], [0],
                       color="#6F7782",
                       linewidth=2.2,
                       label="authored carrier coupling"))
        if event_counts:
            legend.append(
                Line2D([0], [0],
                       color="#7A3E9D",
                       linewidth=2.5,
                       linestyle="--",
                       label="emitted physical event"))
        if legend:
            axis.legend(
                handles=legend,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.03),
                ncol=min(4, len(legend)),
                frameon=False,
                fontsize=8,
            )
        if title:
            axis.set_title(title, fontsize=13, weight="bold", pad=14)
        x_values = tuple(point[0] for point in positions.values())
        y_values = tuple(point[1] for point in positions.values())
        x_span = max(max(x_values) - min(x_values), 1.0)
        y_span = max(max(y_values) - min(y_values), 1.0)
        axis.set_xlim(
            min(x_values) - 0.24 * x_span,
            max(x_values) + 0.24 * x_span,
        )
        axis.set_ylim(
            min(y_values) - 0.30 * y_span,
            max(y_values) + 0.30 * y_span,
        )
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return path


def _encoding_name(encoding) -> str:
    return str(getattr(encoding, "name", encoding))


class MachineGraphView(_GraphView):
    """P1 device-machine spaces, streams, and capability-bearing channels."""

    @classmethod
    def from_device(cls, device) -> "MachineGraphView":
        nodes = []
        for space in device.logical.spaces:
            nodes.append(
                GraphNode(
                    space.name,
                    "region",
                    {
                        "layer":
                            "p1",
                        "label":
                            space.name,
                        "capacity":
                            space.capacity,
                        "capabilities":
                            tuple(capability.key
                                  for capability in space.capabilities),
                        "tags":
                            tuple(space.tags),
                    },
                ))
        for stream in device.logical.streams:
            nodes.append(
                GraphNode(
                    stream.name,
                    "stream",
                    {
                        "layer": "p1",
                        "label": stream.name,
                        "buffer_size": stream.buffer_size,
                        "external": stream.external,
                    },
                ))
        known = {node.id for node in nodes}
        edges = []
        for ordinal, channel in enumerate(device.logical.channels):
            source = getattr(channel.source, "name", None)
            destination = getattr(channel.destination, "name", None)
            if source not in known or destination not in known:
                continue
            edges.append(
                GraphEdge(
                    channel.name or f"channel{ordinal}",
                    "channel",
                    (source, destination),
                    {
                        "capabilities":
                            tuple(capability.key
                                  for capability in channel.capabilities),
                        "direction":
                            str(channel.direction),
                        "concurrency":
                            channel.concurrency,
                    },
                ))
        return cls(
            nodes,
            edges,
            metadata={
                "device": device.name,
                "stage": "p1"
            },
        )


class PatchTopologyView(_GraphView):
    """Static encoded patch slots and their P2-to-P3 carrier realization."""

    @classmethod
    def from_device(cls, device) -> "PatchTopologyView":
        bindings = {
            binding.qec_region.name: binding
            for binding in device.qec_to_physical
        }
        nodes = []
        edges = []
        for region in (() if device.qec is None else device.qec.regions):
            encoding = region.encoding
            binding = bindings.get(region.name)
            topology = binding.patch_topology if binding is not None else None
            capacity = (topology.capacity
                        if topology is not None else region.block_capacity)
            for slot in range(capacity):
                attributes = {
                    "layer": "p2",
                    "label": f"{region.name}[{slot}]",
                    "region": region.name,
                    "slot": slot,
                }
                attributes["encoding"] = _encoding_name(encoding)
                if topology is not None:
                    attributes["carriers"] = topology.carrier_groups[slot]
                    category = topology.categories[slot]
                    if category is not None:
                        attributes["patch_kind"] = category.name
                nodes.append(
                    GraphNode(f"{region.name}:slot{slot}", "patch_slot",
                              attributes))
            if topology is not None:
                for ordinal, (left, right) in enumerate(topology.edges):
                    edges.append(
                        GraphEdge(
                            f"{region.name}.adjacency{ordinal}",
                            "adjacency",
                            (
                                f"{region.name}:slot{left}",
                                f"{region.name}:slot{right}",
                            ),
                            {"carrier_topology": binding.topology.name},
                        ))
        return cls(
            nodes,
            edges,
            metadata={
                "device":
                    device.name,
                "stage": ("p2-p3-binding" if any(
                    binding.patch_topology is not None
                    for binding in bindings.values()) else "p2")
            },
        )


class DeviceStackGraphView(_GraphView):
    """Compact graph of P1 regions, P2 encodings/slots, and P3 resources."""

    @classmethod
    def from_device(cls, device) -> "DeviceStackGraphView":
        nodes = []
        edges = []
        architecture = device.physical
        bindings = {
            binding.qec_region.name: binding
            for binding in device.qec_to_physical
        }
        logical_bindings = {
            binding.logical_region.name: binding
            for binding in device.logical_to_qec
        }
        for space in device.logical.spaces:
            region_id = f"p1:{space.name}"
            nodes.append(
                GraphNode(
                    region_id,
                    "region",
                    {
                        "layer": "p1",
                        "label": space.name,
                        "capacity": space.capacity,
                    },
                ))
            refinement = logical_bindings.get(space.name)
            if refinement is not None:
                qec_region = refinement.qec_region
                encoding_id = f"p2:region:{qec_region.name}"
                edges.append(
                    GraphEdge(
                        f"{space.name}.encoding",
                        "encoded_as",
                        (region_id, encoding_id),
                        {"layer": "binding"},
                    ))

        if device.qec is not None:
            for qec_region in device.qec.regions:
                encoding_id = f"p2:region:{qec_region.name}"
                nodes.append(
                    GraphNode(
                        encoding_id,
                        "qec_region",
                        {
                            "layer": "p2",
                            "label": qec_region.name,
                            "encoding": _encoding_name(qec_region.encoding),
                            "block_capacity": qec_region.block_capacity,
                            "role": qec_region.role,
                        },
                    ))
                binding = bindings.get(qec_region.name)
                topology = (binding.patch_topology
                            if binding is not None else None)
                capacity = (topology.capacity if topology is not None else
                            qec_region.block_capacity)
                for slot in range(capacity):
                    slot_id = f"p2:{qec_region.name}:slot{slot}"
                    category = (topology.categories[slot]
                                if topology is not None else None)
                    nodes.append(
                        GraphNode(
                            slot_id,
                            "patch_slot",
                            {
                                "layer":
                                    "p2",
                                "label":
                                    f"{qec_region.name}[{slot}]",
                                "slot":
                                    slot,
                                "patch_kind": (category.name if category
                                               is not None else None),
                            },
                        ))
                    edges.append(
                        GraphEdge(
                            f"{qec_region.name}.slot{slot}.shape",
                            "has_slot",
                            (encoding_id, slot_id),
                            {"layer": "binding"},
                        ))

        if device.qec is not None:
            for port in device.qec.channel_ports:
                port_id = f"p2:port:{port.name}"
                nodes.append(
                    GraphNode(
                        port_id,
                        "channel_port",
                        {
                            "layer":
                                "p2",
                            "label":
                                port.name,
                            "region":
                                port.region.name,
                            "slot":
                                port.slot,
                            "capabilities":
                                tuple(capability.key
                                      for capability in port.capabilities),
                            "concurrency":
                                port.concurrency,
                            "provider":
                                port.provider,
                            "metadata":
                                dict(port.metadata),
                        },
                    ))
                edges.append(
                    GraphEdge(
                        f"{port.name}.boundary",
                        "exposes",
                        (f"p2:{port.region.name}:slot{port.slot}", port_id),
                        {"layer": "p2"},
                    ))
            for channel in device.qec.channels:
                channel_id = f"p2:channel:{channel.name}"
                nodes.append(
                    GraphNode(
                        channel_id,
                        "qec_channel",
                        {
                            "layer":
                                "p2",
                            "label":
                                channel.name,
                            "capabilities":
                                tuple(capability.key
                                      for capability in channel.capabilities),
                            "concurrency":
                                channel.concurrency,
                            "provider":
                                channel.provider,
                            "protocol": (None if channel.protocol is None else
                                         channel.protocol.name),
                            "metadata":
                                dict(channel.metadata),
                        },
                    ))
                edges.append(
                    GraphEdge(
                        f"{channel_id}.endpoints",
                        "qec_channel",
                        (
                            channel_id,
                            f"p2:port:{channel.source.name}",
                            f"p2:port:{channel.destination.name}",
                        ),
                        {
                            "layer": "p2",
                            "label": channel.name,
                            "concurrency": channel.concurrency,
                            "provider": channel.provider,
                        },
                    ))

        if architecture is not None:
            for resource in architecture.resource_classes:
                nodes.append(
                    GraphNode(
                        f"p3:resource:{resource.name}",
                        "resource_class",
                        {
                            "layer":
                                "p3",
                            "label":
                                f"{resource.name}\n{resource.count} {resource.kind}s",
                            "count":
                                resource.count,
                            "resource_kind":
                                resource.kind,
                        },
                    ))
            for binding in device.qec_to_physical:
                region_id = f"p2:region:{binding.qec_region.name}"
                topology = binding.patch_topology
                if topology is None:
                    for resource in binding.resources:
                        edges.append(
                            GraphEdge(
                                f"{binding.qec_region.name}.{resource.name}",
                                "realized_by",
                                (region_id, f"p3:resource:{resource.name}"),
                                {"layer": "binding"},
                            ))
                    continue
                resource = binding.resources[0]
                for slot, carriers in enumerate(topology.carrier_groups):
                    slot_id = f"p2:{binding.qec_region.name}:slot{slot}"
                    edges.append(
                        GraphEdge(
                            f"{binding.qec_region.name}.slot{slot}.physical",
                            "realized_by",
                            (slot_id, f"p3:resource:{resource.name}"),
                            {
                                "layer": "binding",
                                "carriers": carriers
                            },
                        ))
            for binding in device.qec_channels_to_physical:
                for resource in binding.resources:
                    edges.append(
                        GraphEdge(
                            (f"p2:channel:{binding.qec_channel.name}."
                             f"{resource.name}"),
                            "channel_realized_by",
                            (
                                f"p2:channel:{binding.qec_channel.name}",
                                f"p3:resource:{resource.name}",
                            ),
                            {
                                "layer": "binding",
                                "channel": binding.qec_channel.name,
                            },
                        ))
        return cls(
            nodes,
            edges,
            metadata={
                "device": device.name,
                "stage": "device-stack"
            },
        )


class PatchGraphView(_GraphView):
    """P2 patch instances, interactions, and implicit-slot mapping."""

    @classmethod
    def from_build(cls, build) -> "PatchGraphView | None":
        module = build._fresh_module()
        operations = tuple(view.operation for view in module.body.operations)
        graphs_by_symbol = {
            _text(operation.attributes["sym_name"]): operation
            for operation in operations
            if operation.name == "fabric.patch_graph"
        }
        if not graphs_by_symbol:
            return None
        roots = [
            operation for operation in operations
            if "sym_name" in operation.attributes and
            _text(operation.attributes["sym_name"]) == build.root.symbol
        ]
        if len(roots) != 1:
            raise LookupError(
                "selected Build root is missing or ambiguous while resolving "
                "its patch graph")
        root = roots[0]
        if root.name == "phys.graph" or any(
                attribute in root.attributes
                for attribute in ("graph", "physical_graph")):
            physical_graph = (
                _text(root.attributes["sym_name"]) if root.name == "phys.graph"
                else _text(root.attributes["graph" if "graph" in root.
                                           attributes else "physical_graph"]))
            mappings = [
                operation for operation in operations
                if operation.name == "phys.mapping" and
                _text(operation.attributes["graph"]) == physical_graph
            ]
            if not mappings:
                return None
            if len(mappings) != 1:
                raise LookupError(
                    "selected physical graph has ambiguous patch-graph "
                    "provenance")
            graph_symbol = _text(mappings[0].attributes["source_graph"])
            try:
                graph = graphs_by_symbol[graph_symbol]
            except KeyError as error:
                raise LookupError(
                    "selected physical graph patch provenance is unresolved"
                ) from error
        else:
            selected = [
                operation for operation in graphs_by_symbol.values()
                if _text(operation.attributes["root"]) == build.root.symbol
            ]
            if not selected:
                return None
            if len(selected) != 1:
                raise LookupError(
                    "selected Build root has ambiguous patch graphs")
            graph = selected[0]
        graph_symbol = _text(graph.attributes["sym_name"])
        assignments = {}
        for operation in operations:
            if operation.name != "fabric.patch_mapping":
                continue
            if _text(operation.attributes["graph"]) != graph_symbol:
                continue
            for item in operation.attributes["assignments"]:
                assignment = {
                    "slot": _integer(item["slot"]),
                    "patch_topology": _text(item["topology"]),
                }
                if "carriers" in item:
                    assignment["carrier_nodes"] = tuple(
                        int(value) for value in item["carriers"])
                assignments[_text(item["patch"])] = assignment
        nodes = []
        for item in graph.attributes["nodes"]:
            identity = _text(item["id"])
            attributes = {"code": _text(item["code"])}
            for key in ("encoding", "region", "patch_topology"):
                if key in item:
                    attributes[key] = _text(item[key])
            if "slot" in item:
                attributes["slot"] = _integer(item["slot"])
            attributes.update(assignments.get(identity, {}))
            nodes.append(GraphNode(identity, "patch", attributes))
        edges = []
        for item in graph.attributes["interactions"]:
            edges.append(
                GraphEdge(
                    _text(item["id"]),
                    _text(item["action"]),
                    tuple(_text(value) for value in item["patches"]),
                    {
                        "slots":
                            tuple(int(value) for value in item["slots"])
                            if "slots" in item else (),
                        "carrier_pairs":
                            tuple(
                                tuple(int(value)
                                      for value in pair)
                                for pair in item["pairs"]),
                    },
                ))
        return cls(
            nodes,
            edges,
            metadata={
                "graph": graph_symbol,
                "root": _text(graph.attributes["root"]),
                "stage": "p2",
            },
        )


class CarrierGraphView(_GraphView):
    """P3 device coupling graph with mapped roles and physical-event overlays."""

    @classmethod
    def from_device(cls, device) -> "CarrierGraphView":
        architecture = device.physical
        if architecture is None:
            return cls((), (), metadata={"device": device.name, "stage": "p3"})
        resource_classes = {
            topology.name: {
                resource.name
                for binding in device.qec_to_physical
                if binding.topology is topology
                for resource in binding.resources
            } for topology in architecture.topologies
        }
        membership = {}
        for binding in device.qec_to_physical:
            if binding.topology is None or binding.patch_topology is None:
                continue
            for slot, group in enumerate(binding.patch_topology.carrier_groups):
                category = binding.patch_topology.categories[slot]
                for carrier in group:
                    membership[(binding.topology.name, carrier)] = {
                        "region":
                            binding.qec_region.name,
                        "patch_slot":
                            slot,
                        **({
                            "patch_kind": category.name
                        } if category is not None else {}),
                    }
        nodes = []
        edges = []
        for topology in architecture.topologies:
            for index in topology.nodes:
                identity = f"{topology.name}:{index}"
                nodes.append(
                    GraphNode(
                        identity,
                        "carrier",
                        {
                            "layer":
                                "p3",
                            "topology":
                                topology.name,
                            "index":
                                index,
                            **({
                                "coordinate": topology.coordinates[index]
                            } if topology.coordinates is not None else {}),
                            "resource_classes":
                                tuple(
                                    sorted(
                                        resource_classes.get(topology.name,
                                                             ()))),
                            **membership.get((topology.name, index), {}),
                        },
                    ))
            for ordinal, edge in enumerate(topology.edges):
                edges.append(
                    GraphEdge(
                        f"{topology.name}.coupling{ordinal}",
                        "coupling",
                        (
                            f"{topology.name}:{edge.source}",
                            f"{topology.name}:{edge.target}",
                        ),
                        {"layer": "architecture"},
                    ))
        return cls(
            nodes,
            edges,
            metadata={
                "device": device.name,
                "architecture": architecture.name,
                "stage": "p3",
            },
        )

    @classmethod
    def from_build(cls, build) -> "CarrierGraphView | None":
        module = build._fresh_module()
        operations = tuple(view.operation for view in module.body.operations)
        graph_symbol = build.root.symbol
        root = next(
            (operation for operation in operations
             if "sym_name" in operation.attributes and
             _text(operation.attributes["sym_name"]) == build.root.symbol),
            None,
        )
        if root is not None:
            for attribute in ("graph", "physical_graph"):
                if attribute in root.attributes:
                    graph_symbol = _text(root.attributes[attribute])
                    break
        graph = next(
            (operation for operation in operations
             if operation.name == "phys.graph" and
             _text(operation.attributes["sym_name"]) == graph_symbol),
            None,
        )
        if graph is None:
            return None
        architecture_name = _text(graph.attributes["architecture"])
        architecture = next(
            (operation for operation in operations
             if operation.name == "phys.machine" and
             _text(operation.attributes["sym_name"]) == architecture_name),
            None,
        )
        if architecture is None:
            return None

        topologies = {}
        topology_classes = {}
        patch_membership = {}
        for block in architecture.regions[0].blocks:
            for view in block.operations:
                operation = view.operation
                if (operation.name == "phys.topology" and
                        "num_nodes" in operation.attributes):
                    topologies[_text(
                        operation.attributes["sym_name"])] = operation
                elif operation.name == "phys.patch_topology":
                    patch_name = _text(operation.attributes["sym_name"])
                    carrier_topology = _text(
                        operation.attributes["carrier_topology"])
                    categories = tuple(operation.attributes["categories"])
                    for slot, group in enumerate(
                            operation.attributes["carrier_groups"]):
                        category = _text(categories[slot])
                        for index in group:
                            patch_membership[(carrier_topology, int(index))] = {
                                "patch_topology":
                                    patch_name,
                                "patch_slot":
                                    slot,
                                **({
                                    "patch_kind": category
                                } if category else {}),
                            }
                elif operation.name == "phys.qec_binding" and (
                        "topology" in operation.attributes):
                    topology_name = _text(operation.attributes["topology"])
                    topology_classes.setdefault(topology_name, set()).update(
                        _text(value)
                        for value in operation.attributes["resources"])

        resources = {}
        for operation in operations:
            if operation.name == "phys.resource":
                resources[_text(operation.attributes["sym_name"])] = {
                    "index":
                        _integer(operation.attributes["index"]),
                    "resource_class":
                        _text(operation.attributes["resource_class"]),
                }

        roles_by_resource = {}
        for operation in operations:
            if operation.name != "phys.mapping" or _text(
                    operation.attributes["graph"]) != graph_symbol:
                continue
            for item in operation.attributes["initial"]:
                roles_by_resource.setdefault(_text(item["resource"]),
                                             []).append(_text(item["role"]))

        nodes = []
        edges = []
        node_id = {}
        for topology_name, topology in sorted(topologies.items()):
            bound_classes = topology_classes.get(topology_name, set())
            coordinates = (tuple(
                tuple(map(int, coordinate))
                for coordinate in topology.attributes["coordinates"])
                           if "coordinates" in topology.attributes else ())
            for index in range(_integer(topology.attributes["num_nodes"])):
                identity = f"{topology_name}:{index}"
                node_id[(topology_name, index)] = identity
                carrier_roles = []
                carrier_classes = set()
                for symbol, resource in resources.items():
                    if resource["index"] != index or (bound_classes and
                                                      resource["resource_class"]
                                                      not in bound_classes):
                        continue
                    carrier_classes.add(resource["resource_class"])
                    carrier_roles.extend(roles_by_resource.get(symbol, ()))
                nodes.append(
                    GraphNode(
                        identity,
                        "carrier",
                        {
                            "topology":
                                topology_name,
                            "index":
                                index,
                            **({
                                "coordinate": coordinates[index]
                            } if coordinates else {}),
                            "resource_classes":
                                tuple(sorted(carrier_classes)),
                            "roles":
                                tuple(sorted(carrier_roles)),
                            **patch_membership.get((topology_name, index), {}),
                        },
                    ))
            topology_edges = (topology.attributes["edges"]
                              if "edges" in topology.attributes else ())
            for ordinal, item in enumerate(topology_edges):
                source, target = map(int, item)
                edges.append(
                    GraphEdge(
                        f"{topology_name}.coupling{ordinal}",
                        "coupling",
                        (
                            node_id[(topology_name, source)],
                            node_id[(topology_name, target)],
                        ),
                        {"layer": "architecture"},
                    ))

        for operation in _walk(graph):
            if operation.name != "phys.apply" or "resources" not in operation.attributes:
                continue
            symbols = tuple(
                _text(value) for value in operation.attributes["resources"])
            if len(symbols) < 2:
                continue
            topology_name = (_text(operation.attributes["topology"])
                             if "topology" in operation.attributes else next(
                                 iter(topologies), None))
            if topology_name is None:
                continue
            endpoints = tuple(node_id[(topology_name,
                                       resources[symbol]["index"])]
                              for symbol in symbols)
            edges.append(
                GraphEdge(
                    _text(operation.attributes["event_id"]) if "event_id"
                    in operation.attributes else f"event{len(edges)}",
                    _text(operation.attributes["action"]),
                    endpoints,
                    {
                        "layer": "event",
                        "resources": symbols
                    },
                ))
        return cls(
            nodes,
            edges,
            metadata={
                "graph": build.root.symbol,
                "architecture": architecture_name,
                "stage": "p3",
            },
        )


__all__ = [
    "GraphNode",
    "GraphEdge",
    "MachineGraphView",
    "PatchTopologyView",
    "DeviceStackGraphView",
    "PatchGraphView",
    "CarrierGraphView",
]
