# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeline visualization for pulse schedules.

Plots a per-line timeline of drive, readout, wait, and sync events
using matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from ..passes.ir_types import Program
from ..passes.scheduling import ScheduledEvent

_DRIVE_COLOR = "#4C72B0"
_READOUT_COLOR = "#DD8452"
_WAIT_COLOR = "#CCCCCC"
_SYNC_COLOR = "#55A868"
_TONE_MOD_COLOR = "#C44E52"


def plot_schedule(
    events: list[ScheduledEvent],
    program: Program | None = None,
    figsize: tuple[float, float] = (14, 4),
    title: str | None = None,
) -> Figure:
    """Plot a pulse schedule timeline.

    Args:
        events: Scheduled events from the scheduling pass.
        program: Optional program for additional metadata.
        figsize: Figure size (width, height).
        title: Optional figure title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    line_ids: list[int] = []
    for ev in events:
        if ev.line_id is not None and ev.line_id not in line_ids:
            line_ids.append(ev.line_id)

    line_to_row = {lid: i for i, lid in enumerate(line_ids)}
    n_rows = max(len(line_ids), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for ev in events:
        if ev.line_id is None:
            continue
        row = line_to_row.get(ev.line_id)
        if row is None:
            continue

        y = row
        x = ev.start_vtu
        w = ev.duration_vtu

        if ev.kind in ("drive",):
            rect = mpatches.FancyBboxPatch(
                (x, y - 0.35),
                w,
                0.7,
                boxstyle="round,pad=0.02",
                facecolor=_DRIVE_COLOR,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )
            ax.add_patch(rect)
        elif ev.kind in ("readout", "iq_acquire"):
            rect = mpatches.FancyBboxPatch(
                (x, y - 0.35),
                w,
                0.7,
                boxstyle="round,pad=0.02",
                facecolor=_READOUT_COLOR,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )
            ax.add_patch(rect)
        elif ev.kind == "wait" and w > 0:
            rect = mpatches.Rectangle(
                (x, y - 0.3),
                w,
                0.6,
                facecolor=_WAIT_COLOR,
                edgecolor="gray",
                linewidth=0.3,
                alpha=0.5,
                hatch="//",
            )
            ax.add_patch(rect)

    sync_times: set[int] = set()
    for ev in events:
        if ev.kind == "sync":
            sync_times.add(ev.start_vtu)
    for t in sync_times:
        ax.axvline(x=t,
                   color=_SYNC_COLOR,
                   linestyle="--",
                   linewidth=1,
                   alpha=0.7)

    for ev in events:
        if ev.kind in ("shift_phase", "set_phase") and ev.tone_id is not None:
            for lid, row in line_to_row.items():
                ax.plot(
                    ev.start_vtu,
                    row,
                    marker="^",
                    color=_TONE_MOD_COLOR,
                    markersize=6,
                    zorder=5,
                )
                break
        elif ev.kind in ("shift_frequency",
                         "set_frequency") and ev.tone_id is not None:
            for lid, row in line_to_row.items():
                ax.plot(
                    ev.start_vtu,
                    row,
                    marker="s",
                    color=_TONE_MOD_COLOR,
                    markersize=5,
                    zorder=5,
                )
                break

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"line {lid}" for lid in line_ids])
    ax.set_xlabel("Time (VTU)")
    ax.set_ylim(-0.5, n_rows - 0.5)

    max_time = max((ev.start_vtu + ev.duration_vtu for ev in events),
                   default=100)
    ax.set_xlim(-max_time * 0.02, max_time * 1.05)

    if title:
        ax.set_title(title)
    elif program:
        ax.set_title(f"Pulse Schedule: {program.name}")

    legend_handles = [
        mpatches.Patch(color=_DRIVE_COLOR, label="Drive"),
        mpatches.Patch(color=_READOUT_COLOR, label="Readout"),
        mpatches.Patch(color=_WAIT_COLOR, label="Wait", hatch="//"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def save_schedule(
    events: list[ScheduledEvent],
    path: str,
    program: Program | None = None,
    **kwargs: Any,
) -> None:
    """Plot and save a schedule to a file."""
    fig = plot_schedule(events, program=program, **kwargs)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
