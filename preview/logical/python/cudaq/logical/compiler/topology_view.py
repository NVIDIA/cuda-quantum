# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed, read-only graph views over canonical LVM and Fabric IR facts."""

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

        lines = ["graph logical {"]
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
        authored carrier-architecture layer.
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
        }

        width = max(7.0, min(13.0, 1.0 + 0.85 * len(self.nodes)))
        height = max(4.2, min(8.5, 2.8 + 0.36 * len(self.nodes)))
        figure, axis = plt.subplots(figsize=(width, height))
        axis.set_facecolor("#FAFAFA")

        for edge in self.edges:
            anchor = edge.endpoints[0]
            for endpoint in edge.endpoints[1:]:
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
            present_layers = tuple(layer for layer in ("p1", "p2") if any(
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
        for ordinal, channel in enumerate(device.logical._channels):
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


class PatchGraphView(_GraphView):
    """P2 patch instances and logical interactions."""

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
        selected = [
            operation for operation in graphs_by_symbol.values()
            if _text(operation.attributes["root"]) == build.root.symbol
        ]
        if not selected:
            return None
        if len(selected) != 1:
            raise LookupError("selected Build root has ambiguous patch graphs")
        graph = selected[0]
        graph_symbol = _text(graph.attributes["sym_name"])
        nodes = []
        for item in graph.attributes["nodes"]:
            identity = _text(item["id"])
            attributes = {"code": _text(item["code"])}
            for key in ("encoding", "region", "patch_topology"):
                if key in item:
                    attributes[key] = _text(item[key])
            if "slot" in item:
                attributes["slot"] = _integer(item["slot"])
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


__all__ = [
    "GraphNode",
    "GraphEdge",
    "MachineGraphView",
    "PatchGraphView",
]
