# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic bug corpus: 100 programs across 4 bug classes + positive controls.

Bug classes (paper Section 5): unintentional_overlap, backward_time_travel,
phase_bookkeeping, cross_resonance_miscalibration.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Any
import cudaq_pulse as pulse


@dataclass
class BugCase:
    name: str
    expected: str
    build_fn: Callable[[], Any]
    note: str = ""


# -- shared helpers used inside kernels -------------------------------------
def _g():
    return gaussian(40, 0.1, 10.0)


def _x():
    return square(40, [0.047, 0.0])


def _cr():
    return gaussian(200, 0.10, 50.0)


def _crn():
    return gaussian(200, -0.10, 50.0)


def _sx():
    return drag(40, 0.025, 10.0, 0.5)


def _ro():
    return square(1000, [0.05, 0.0])


def _C(n, fn, note=""):
    return BugCase(n, "correct", fn, note)


def _O(n, fn, note=""):
    return BugCase(n, "unintentional_overlap", fn, note)


def _T(n, fn, note=""):
    return BugCase(n, "backward_time_travel", fn, note)


def _P(n, fn, note=""):
    return BugCase(n, "phase_bookkeeping", fn, note)


def _R(n, fn, note=""):
    return BugCase(n, "cross_resonance_miscalibration", fn, note)


# ═══ Correct (25) ═════════════════════════════════════════════════════════
def _c_dd(n):

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        x = _x()
        for _ in range(n):
            drive(d, x, t)
            wait(d, 200)

    return k


def _c_1q(body_tag):
    """Factory for 1-qubit correct patterns."""

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        if body_tag == "drive":
            drive(d, _g(), t)
        elif body_tag == "seq4":
            for _ in range(4):
                drive(d, _g(), t)
        elif body_tag == "wait_d":
            drive(d, _g(), t)
            wait(d, 100)
            drive(d, _g(), t)
        elif body_tag == "phase":
            shift_phase(t, 0.5)
            drive(d, _g(), t)
        elif body_tag == "vz":
            drive(d, _g(), t)
            shift_phase(t, math.pi)
            drive(d, _g(), t)
        elif body_tag == "zph":
            shift_phase(t, 0.0)
            drive(d, _g(), t)
        elif body_tag == "fdet":
            shift_frequency(t, 1e6)
            drive(d, _g(), t)
            shift_frequency(t, -1e6)
        elif body_tag == "wait":
            wait(d, 500)
        elif body_tag == "lwait":
            wait(d, 10000)
            drive(d, _g(), t)
        elif body_tag == "sx4":
            sx = drag(40, 0.25, 10.0, 0.5)
            for _ in range(4):
                shift_phase(t, math.pi / 2)
                drive(d, sx, t)

    return k


CORRECT = [
    _C(f"c_{t}", lambda t=t: _c_1q(t), t) for t in [
        "drive", "seq4", "wait_d", "phase", "vz", "zph", "fdet", "wait",
        "lwait", "sx4"
    ]
]
CORRECT += [
    _C(f"c_dd{n}", lambda n=n: _c_dd(n), f"DD-{n}")
    for n in [1, 2, 3, 5, 8, 10, 16, 20, 50, 100]
]


@pulse.kernel
def _c2q(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    wf = _g()
    drive(d0, wf, t0)
    drive(d1, wf, t1)


@pulse.kernel
def _cscr(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d0, _cr(), t1)


@pulse.kernel
def _cro2(q0, q1):
    r0, rt0 = get_readout_line(q0)
    r1, rt1 = get_readout_line(q1)
    ro = _ro()
    readout(r0, ro, rt0, "iq")
    readout(r1, ro, rt1, "iq")


@pulse.kernel
def _cmux(q):
    d, t = get_drive_line(q)
    drive(d, wf_add(gaussian(80, 0.05, 20.), gaussian(80, 0.04, 15.)), t)


@pulse.kernel
def _c3s(q0, q1, q2):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    drive(d0, _g(), t0)
    sync(d0, d1, d2)
    drive(d1, _g(), t1)


CORRECT += [
    _C("c_2qpar", lambda: _c2q),
    _C("c_syncr", lambda: _cscr),
    _C("c_ro2q", lambda: _cro2),
    _C("c_mux", lambda: _cmux),
    _C("c_3sync", lambda: _c3s)
]


# ═══ Overlap (25) ═════════════════════════════════════════════════════════
def _b_ovk(k):

    @pulse.kernel
    def kern(q):
        d, t = get_drive_line(q)
        wf = _g()
        for _ in range(k):
            drive(d, wf, t)  # BUG: line not rebound

    return kern


@pulse.kernel
def _b_ovcr(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d0, _cr(), t1)
    drive(d0, _x(), t0)  # BUG


@pulse.kernel
def _b_ovro(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, gaussian(200, 0.1, 50.), t)
    readout(r, _ro(), rt, "iq")  # BUG: no sync


OVERLAP = [
    _O(f"b_ov{n}", lambda n=n: _b_ovk(n), f"{n}x") for n in [
        2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 80, 96, 100, 128, 150,
        200, 250, 300, 512
    ]
]
OVERLAP += [_O("b_ovcr", lambda: _b_ovcr), _O("b_ovro", lambda: _b_ovro)]


# ═══ Backward time-travel (20) ════════════════════════════════════════════
def _b_neg(dur):

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, _g(), t)
        wait(d, dur)

    return k


def _b_negpre(dur):

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        wait(d, dur)
        drive(d, _g(), t)

    return k


def _b_negro(dur):

    @pulse.kernel
    def k(q):
        r, rt = get_readout_line(q)
        wait(r, dur)
        readout(r, _ro(), rt, "iq")

    return k


def _b_negbtw(dur):

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, _g(), t)
        wait(d, dur)
        drive(d, _g(), t)

    return k


BTT = [
    _T(f"b_neg{v}", lambda v=v: _b_neg(-v), f"wait {-v}")
    for v in [1, 5, 10, 40, 100, 500, 1000]
]
BTT += [
    _T(f"b_negp{v}", lambda v=v: _b_negpre(-v), f"pre {-v}")
    for v in [1, 10, 100, 1000]
]
BTT += [
    _T(f"b_negr{v}", lambda v=v: _b_negro(-v), f"ro {-v}")
    for v in [1, 10, 100, 500]
]
BTT += [
    _T(f"b_negb{v}", lambda v=v: _b_negbtw(-v), f"btw {-v}")
    for v in [10, 40, 100, 200, 1000]
]


# ═══ Phase bookkeeping (15) ═══════════════════════════════════════════════
@pulse.kernel
def _b_phst(q):
    d, t = get_drive_line(q)
    drive(d, _g(), t)
    shift_phase(t, 1.0)


@pulse.kernel
def _b_phdbl(q):
    d, t = get_drive_line(q)
    shift_phase(t, 0.5)
    shift_phase(t, 0.3)


@pulse.kernel
def _b_phcr(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d0, _cr(), t1)
    shift_phase(t1, math.pi / 2)


@pulse.kernel
def _b_phfr(q):
    d, t = get_drive_line(q)
    drive(d, _g(), t)
    shift_frequency(t, 1e6)


def _b_phk(n):

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        shift_phase(t, 0.1)
        for _ in range(n):
            shift_phase(t, 0.1)

    return k


PHASE = [
    _P("b_phst", lambda: _b_phst),
    _P("b_phdbl", lambda: _b_phdbl),
    _P("b_phcr", lambda: _b_phcr),
    _P("b_phfr", lambda: _b_phfr)
]
PHASE += [
    _P(f"b_phk{n}", lambda n=n: _b_phk(n), f"{n} stale")
    for n in [2, 3, 4, 5, 6, 8, 10, 16, 20, 32, 64]
]


# ═══ CR miscalibration (15) ═══════════════════════════════════════════════
@pulse.kernel
def _b_crt(q0, q1):  # wrong tone
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d0, _cr(), t0)


@pulse.kernel
def _b_crne(q0, q1):  # no echo
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d1, _sx(), t1)
    drive(d0, _cr(), t1)
    drive(d0, _crn(), t1)
    drive(d1, _sx(), t1)


@pulse.kernel
def _b_crsw(q0, q1):  # swapped line
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d1, _cr(), t1)


@pulse.kernel
def _b_crns(q0, q1):  # no sync
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    drive(d1, _sx(), t1)
    drive(d0, _cr(), t1)
    drive(d0, _x(), t0)
    drive(d0, _crn(), t1)
    drive(d1, _sx(), t1)


@pulse.kernel
def _b_crde(q0, q1):  # double echo
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d1, _sx(), t1)
    drive(d0, _cr(), t1)
    drive(d0, _x(), t0)
    drive(d0, _x(), t0)
    drive(d0, _crn(), t1)
    drive(d1, _sx(), t1)


@pulse.kernel
def _b_cram(q0, q1):  # asymmetric amp
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d1, _sx(), t1)
    drive(d0, _cr(), t1)
    drive(d0, _x(), t0)
    drive(d0, gaussian(200, -0.08, 50.), t1)
    drive(d1, _sx(), t1)


@pulse.kernel
def _b_crboth(q0, q1):  # both wrong tone
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d1, _sx(), t1)
    drive(d0, _cr(), t0)
    drive(d0, _x(), t0)
    drive(d0, _crn(), t0)
    drive(d1, _sx(), t1)


@pulse.kernel
def _b_crrev(q0, q1):  # reversed order
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d1, _sx(), t1)
    drive(d0, _crn(), t1)
    drive(d0, _x(), t0)
    drive(d0, _cr(), t1)
    drive(d1, _sx(), t1)


@pulse.kernel
def _b_crnsx(q0, q1):  # no SX on target
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d0, _cr(), t1)
    drive(d0, _x(), t0)
    drive(d0, _crn(), t1)


@pulse.kernel
def _b_crself(q0, q1):  # self-drive
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d0, _cr(), t0)
    drive(d1, _cr(), t1)


@pulse.kernel
def _b_cr3q(q0, q1, q2):  # 3-qubit mixup
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    sync(d0, d1, d2)
    drive(d0, _cr(), t2)
    drive(d1, _cr(), t0)


@pulse.kernel
def _b_crex(q0, q1):  # extra spurious
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d1, _sx(), t1)
    drive(d0, _cr(), t1)
    drive(d1, gaussian(40, 0.01, 10.), t1)
    drive(d0, _x(), t0)
    drive(d0, _crn(), t1)
    drive(d1, _sx(), t1)


@pulse.kernel
def _b_crsig(q0, q1):  # sigma too small
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d0, gaussian(200, 0.10, 5.0), t1)


@pulse.kernel
def _b_crlong(q0, q1):  # too long
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d0, gaussian(2000, 0.10, 500.), t1)


@pulse.kernel
def _b_crdur(q0, q1):  # CR+/CR- duration mismatch
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    drive(d1, _sx(), t1)
    drive(d0, _cr(), t1)
    drive(d0, _x(), t0)
    drive(d0, gaussian(160, -0.10, 40.), t1)
    drive(d1, _sx(), t1)


CR = [
    _R("b_crt", lambda: _b_crt),
    _R("b_crne", lambda: _b_crne),
    _R("b_crsw", lambda: _b_crsw),
    _R("b_crns", lambda: _b_crns),
    _R("b_crde", lambda: _b_crde),
    _R("b_cram", lambda: _b_cram),
    _R("b_crboth", lambda: _b_crboth),
    _R("b_crrev", lambda: _b_crrev),
    _R("b_crnsx", lambda: _b_crnsx),
    _R("b_crself", lambda: _b_crself),
    _R("b_cr3q", lambda: _b_cr3q),
    _R("b_crex", lambda: _b_crex),
    _R("b_crsig", lambda: _b_crsig),
    _R("b_crlong", lambda: _b_crlong),
    _R("b_crdur", lambda: _b_crdur)
]

# ═══ Full corpus ══════════════════════════════════════════════════════════
ALL_CASES: list[BugCase] = CORRECT + OVERLAP + BTT + PHASE + CR
CORPUS_STATS = {
    c: sum(1 for x in ALL_CASES if x.expected == c) for c in [
        "correct", "unintentional_overlap", "backward_time_travel",
        "phase_bookkeeping", "cross_resonance_miscalibration"
    ]
}
CORPUS_STATS["total"] = len(ALL_CASES)

if __name__ == "__main__":
    for c in ALL_CASES:
        try:
            c.build_fn()
            s = "OK"
        except Exception as e:
            s = f"FAIL: {e}"
        print(f"{c.name:18s} [{c.expected:40s}] {s}")
    print(f"\n{CORPUS_STATS}")
