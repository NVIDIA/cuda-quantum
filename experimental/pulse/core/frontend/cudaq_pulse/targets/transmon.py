# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transmon QPU target definitions.

The default target is the 17-qubit device from:
    Krinner et al., "Realizing Repeated Quantum Error Correction in a
    Distance-Three Surface Code", PRX 12, 021049 (2022).
    arXiv:2112.03708

Device parameters extracted from calibration data in the cudaq-qlx-cal
repository (Dialect/Cal/Krinner/*.mlir).
"""

from __future__ import annotations

from .base import Coupling, CrosstalkEntry, Qubit, Target


def transmon_krinner_17q() -> Target:
    """17-qubit superconducting transmon target (Krinner et al. 2022).

    Layout (rotated d=3 surface code):
        D1(0) -- Z1(9) -- D2(1)
        |                  |
        X1(13)            X2(14) -- D3(2)
        |                  |
        D4(3) -- Z2(10)-- D5(4) -- Z3(11)-- D6(5)
        |                  |                  |
                  X3(15)            X4(16)
        |                  |                  |
        D7(6) ----------- D8(7) -- Z4(12)-- D9(8)

    9 data qubits (D1-D9, idx 0-8), 4 Z-ancillas (idx 9-12),
    4 X-ancillas (idx 13-16). Native 2Q gate: CZ (98 ns).
    """

    qubits = {
        # --- Data qubits D1-D9 (indices 0-8) ---
        0:
            Qubit(index=0,
                  frequency_hz=5.100e9,
                  anharmonicity_hz=-330e6,
                  t1_us=24.0,
                  t2_star_us=15.0,
                  label="D1",
                  drive_params={
                      "x_amp": 0.432,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.73
                  },
                  readout_params={
                      "ro_freq": 6.850e9,
                      "ro_amp": 0.185,
                      "ro_dur": 600.0
                  }),
        1:
            Qubit(index=1,
                  frequency_hz=5.210e9,
                  anharmonicity_hz=-325e6,
                  t1_us=28.0,
                  t2_star_us=20.0,
                  label="D2",
                  drive_params={
                      "x_amp": 0.445,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.68
                  },
                  readout_params={
                      "ro_freq": 6.920e9,
                      "ro_amp": 0.190,
                      "ro_dur": 600.0
                  }),
        2:
            Qubit(index=2,
                  frequency_hz=5.050e9,
                  anharmonicity_hz=-335e6,
                  t1_us=21.0,
                  t2_star_us=8.0,
                  label="D3",
                  drive_params={
                      "x_amp": 0.438,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.71
                  },
                  readout_params={
                      "ro_freq": 6.980e9,
                      "ro_amp": 0.200,
                      "ro_dur": 600.0
                  }),
        3:
            Qubit(index=3,
                  frequency_hz=5.150e9,
                  anharmonicity_hz=-328e6,
                  t1_us=26.0,
                  t2_star_us=17.0,
                  label="D4",
                  drive_params={
                      "x_amp": 0.441,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.69
                  },
                  readout_params={
                      "ro_freq": 7.040e9,
                      "ro_amp": 0.192,
                      "ro_dur": 600.0
                  }),
        4:
            Qubit(index=4,
                  frequency_hz=5.180e9,
                  anharmonicity_hz=-332e6,
                  t1_us=19.0,
                  t2_star_us=5.0,
                  label="D5",
                  drive_params={
                      "x_amp": 0.435,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.75
                  },
                  readout_params={
                      "ro_freq": 7.100e9,
                      "ro_amp": 0.188,
                      "ro_dur": 600.0
                  }),
        5:
            Qubit(index=5,
                  frequency_hz=5.090e9,
                  anharmonicity_hz=-327e6,
                  t1_us=27.0,
                  t2_star_us=18.0,
                  label="D6",
                  drive_params={
                      "x_amp": 0.448,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.72
                  },
                  readout_params={
                      "ro_freq": 7.150e9,
                      "ro_amp": 0.210,
                      "ro_dur": 600.0
                  }),
        6:
            Qubit(index=6,
                  frequency_hz=5.220e9,
                  anharmonicity_hz=-331e6,
                  t1_us=23.0,
                  t2_star_us=12.0,
                  label="D7",
                  drive_params={
                      "x_amp": 0.429,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.70
                  },
                  readout_params={
                      "ro_freq": 6.880e9,
                      "ro_amp": 0.195,
                      "ro_dur": 600.0
                  }),
        7:
            Qubit(index=7,
                  frequency_hz=5.130e9,
                  anharmonicity_hz=-329e6,
                  t1_us=31.0,
                  t2_star_us=23.0,
                  label="D8",
                  drive_params={
                      "x_amp": 0.440,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.74
                  },
                  readout_params={
                      "ro_freq": 6.960e9,
                      "ro_amp": 0.205,
                      "ro_dur": 600.0
                  }),
        8:
            Qubit(index=8,
                  frequency_hz=5.070e9,
                  anharmonicity_hz=-334e6,
                  t1_us=25.0,
                  t2_star_us=14.0,
                  label="D9",
                  drive_params={
                      "x_amp": 0.436,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.67
                  },
                  readout_params={
                      "ro_freq": 7.020e9,
                      "ro_amp": 0.187,
                      "ro_dur": 600.0
                  }),
        # --- Z-ancillas Z1-Z4 (indices 9-12) ---
        9:
            Qubit(index=9,
                  frequency_hz=4.250e9,
                  anharmonicity_hz=-340e6,
                  t1_us=18.0,
                  t2_star_us=6.0,
                  label="Z1",
                  drive_params={
                      "x_amp": 0.452,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.66
                  },
                  readout_params={
                      "ro_freq": 6.810e9,
                      "ro_amp": 0.198,
                      "ro_dur": 600.0
                  }),
        10:
            Qubit(index=10,
                  frequency_hz=4.320e9,
                  anharmonicity_hz=-338e6,
                  t1_us=22.0,
                  t2_star_us=10.0,
                  label="Z2",
                  drive_params={
                      "x_amp": 0.447,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.71
                  },
                  readout_params={
                      "ro_freq": 6.870e9,
                      "ro_amp": 0.202,
                      "ro_dur": 600.0
                  }),
        11:
            Qubit(index=11,
                  frequency_hz=4.410e9,
                  anharmonicity_hz=-336e6,
                  t1_us=20.0,
                  t2_star_us=7.0,
                  label="Z3",
                  drive_params={
                      "x_amp": 0.443,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.69
                  },
                  readout_params={
                      "ro_freq": 6.940e9,
                      "ro_amp": 0.191,
                      "ro_dur": 600.0
                  }),
        12:
            Qubit(index=12,
                  frequency_hz=4.480e9,
                  anharmonicity_hz=-342e6,
                  t1_us=17.0,
                  t2_star_us=2.0,
                  label="Z4",
                  drive_params={
                      "x_amp": 0.439,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.74
                  },
                  readout_params={
                      "ro_freq": 7.010e9,
                      "ro_amp": 0.208,
                      "ro_dur": 600.0
                  }),
        # --- X-ancillas X1-X4 (indices 13-16) ---
        13:
            Qubit(index=13,
                  frequency_hz=4.620e9,
                  anharmonicity_hz=-337e6,
                  t1_us=17.0,
                  t2_star_us=3.0,
                  label="X1",
                  drive_params={
                      "x_amp": 0.450,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.68
                  },
                  readout_params={
                      "ro_freq": 7.080e9,
                      "ro_amp": 0.186,
                      "ro_dur": 600.0
                  }),
        14:
            Qubit(index=14,
                  frequency_hz=4.710e9,
                  anharmonicity_hz=-339e6,
                  t1_us=23.0,
                  t2_star_us=13.0,
                  label="X2",
                  drive_params={
                      "x_amp": 0.446,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.72
                  },
                  readout_params={
                      "ro_freq": 7.130e9,
                      "ro_amp": 0.215,
                      "ro_dur": 600.0
                  }),
        15:
            Qubit(index=15,
                  frequency_hz=4.830e9,
                  anharmonicity_hz=-335e6,
                  t1_us=19.0,
                  t2_star_us=9.0,
                  label="X3",
                  drive_params={
                      "x_amp": 0.441,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.70
                  },
                  readout_params={
                      "ro_freq": 7.190e9,
                      "ro_amp": 0.194,
                      "ro_dur": 600.0
                  }),
        16:
            Qubit(index=16,
                  frequency_hz=4.900e9,
                  anharmonicity_hz=-341e6,
                  t1_us=18.0,
                  t2_star_us=4.0,
                  label="X4",
                  drive_params={
                      "x_amp": 0.434,
                      "x_dur": 20.0,
                      "x_sigma": 5.0,
                      "x_beta": 0.75
                  },
                  readout_params={
                      "ro_freq": 7.060e9,
                      "ro_amp": 0.220,
                      "ro_dur": 600.0
                  }),
    }

    # 24 CZ coupling edges (ancilla <-> data pairs)
    couplings = [
        # Z1 (wt-2): D1-Z1, D2-Z1
        Coupling(0,
                 9,
                 coupling_strength_hz=3.20e6,
                 gate_fidelity=0.986,
                 gate_params={
                     "cz_amp": 0.320,
                     "cz_phase_correction": 0.045
                 }),
        Coupling(1,
                 9,
                 coupling_strength_hz=3.10e6,
                 gate_fidelity=0.984,
                 gate_params={
                     "cz_amp": 0.310,
                     "cz_phase_correction": 0.052
                 }),
        # Z2 (wt-4): D1-Z2, D2-Z2, D4-Z2, D5-Z2
        Coupling(0,
                 10,
                 coupling_strength_hz=3.35e6,
                 gate_fidelity=0.985,
                 gate_params={
                     "cz_amp": 0.335,
                     "cz_phase_correction": 0.038
                 }),
        Coupling(1,
                 10,
                 coupling_strength_hz=3.28e6,
                 gate_fidelity=0.987,
                 gate_params={
                     "cz_amp": 0.328,
                     "cz_phase_correction": 0.041
                 }),
        Coupling(3,
                 10,
                 coupling_strength_hz=3.40e6,
                 gate_fidelity=0.983,
                 gate_params={
                     "cz_amp": 0.340,
                     "cz_phase_correction": 0.063
                 }),
        Coupling(4,
                 10,
                 coupling_strength_hz=3.15e6,
                 gate_fidelity=0.986,
                 gate_params={
                     "cz_amp": 0.315,
                     "cz_phase_correction": 0.057
                 }),
        # Z3 (wt-4): D5-Z3, D6-Z3, D8-Z3, D9-Z3
        Coupling(4,
                 11,
                 coupling_strength_hz=3.30e6,
                 gate_fidelity=0.985,
                 gate_params={
                     "cz_amp": 0.330,
                     "cz_phase_correction": 0.049
                 }),
        Coupling(5,
                 11,
                 coupling_strength_hz=3.45e6,
                 gate_fidelity=0.982,
                 gate_params={
                     "cz_amp": 0.345,
                     "cz_phase_correction": 0.071
                 }),
        Coupling(7,
                 11,
                 coupling_strength_hz=3.18e6,
                 gate_fidelity=0.988,
                 gate_params={
                     "cz_amp": 0.318,
                     "cz_phase_correction": 0.035
                 }),
        Coupling(8,
                 11,
                 coupling_strength_hz=3.25e6,
                 gate_fidelity=0.984,
                 gate_params={
                     "cz_amp": 0.325,
                     "cz_phase_correction": 0.055
                 }),
        # Z4 (wt-2): D8-Z4, D9-Z4
        Coupling(7,
                 12,
                 coupling_strength_hz=3.10e6,
                 gate_fidelity=0.986,
                 gate_params={
                     "cz_amp": 0.310,
                     "cz_phase_correction": 0.043
                 }),
        Coupling(8,
                 12,
                 coupling_strength_hz=3.38e6,
                 gate_fidelity=0.983,
                 gate_params={
                     "cz_amp": 0.338,
                     "cz_phase_correction": 0.067
                 }),
        # X1 (wt-2): D1-X1, D4-X1
        Coupling(0,
                 13,
                 coupling_strength_hz=3.05e6,
                 gate_fidelity=0.987,
                 gate_params={
                     "cz_amp": 0.305,
                     "cz_phase_correction": 0.059
                 }),
        Coupling(3,
                 13,
                 coupling_strength_hz=3.42e6,
                 gate_fidelity=0.984,
                 gate_params={
                     "cz_amp": 0.342,
                     "cz_phase_correction": 0.047
                 }),
        # X2 (wt-4): D2-X2, D3-X2, D5-X2, D6-X2
        Coupling(1,
                 14,
                 coupling_strength_hz=3.27e6,
                 gate_fidelity=0.985,
                 gate_params={
                     "cz_amp": 0.327,
                     "cz_phase_correction": 0.061
                 }),
        Coupling(2,
                 14,
                 coupling_strength_hz=3.13e6,
                 gate_fidelity=0.988,
                 gate_params={
                     "cz_amp": 0.313,
                     "cz_phase_correction": 0.032
                 }),
        Coupling(4,
                 14,
                 coupling_strength_hz=3.36e6,
                 gate_fidelity=0.982,
                 gate_params={
                     "cz_amp": 0.336,
                     "cz_phase_correction": 0.074
                 }),
        Coupling(5,
                 14,
                 coupling_strength_hz=3.22e6,
                 gate_fidelity=0.986,
                 gate_params={
                     "cz_amp": 0.322,
                     "cz_phase_correction": 0.050
                 }),
        # X3 (wt-4): D4-X3, D5-X3, D7-X3, D8-X3
        Coupling(3,
                 15,
                 coupling_strength_hz=3.08e6,
                 gate_fidelity=0.987,
                 gate_params={
                     "cz_amp": 0.308,
                     "cz_phase_correction": 0.040
                 }),
        Coupling(4,
                 15,
                 coupling_strength_hz=3.47e6,
                 gate_fidelity=0.981,
                 gate_params={
                     "cz_amp": 0.347,
                     "cz_phase_correction": 0.082
                 }),
        Coupling(6,
                 15,
                 coupling_strength_hz=3.19e6,
                 gate_fidelity=0.986,
                 gate_params={
                     "cz_amp": 0.319,
                     "cz_phase_correction": 0.054
                 }),
        Coupling(7,
                 15,
                 coupling_strength_hz=3.31e6,
                 gate_fidelity=0.984,
                 gate_params={
                     "cz_amp": 0.331,
                     "cz_phase_correction": 0.068
                 }),
        # X4 (wt-2): D6-X4, D9-X4
        Coupling(5,
                 16,
                 coupling_strength_hz=3.24e6,
                 gate_fidelity=0.988,
                 gate_params={
                     "cz_amp": 0.324,
                     "cz_phase_correction": 0.036
                 }),
        Coupling(8,
                 16,
                 coupling_strength_hz=3.06e6,
                 gate_fidelity=0.983,
                 gate_params={
                     "cz_amp": 0.306,
                     "cz_phase_correction": 0.078
                 }),
    ]

    crosstalk = [
        CrosstalkEntry(0,
                       5,
                       zz_coupling=8.5e-4,
                       static_zz_hz=1.2e5,
                       freq_delta_hz=1.0e7),
        CrosstalkEntry(2,
                       8,
                       zz_coupling=6.2e-4,
                       static_zz_hz=8.5e4,
                       freq_delta_hz=2.0e7),
        CrosstalkEntry(4,
                       3,
                       zz_coupling=4.8e-4,
                       static_zz_hz=6.2e4,
                       freq_delta_hz=3.0e7),
        CrosstalkEntry(1,
                       6,
                       zz_coupling=9.1e-4,
                       static_zz_hz=1.35e5,
                       freq_delta_hz=1.0e7),
        CrosstalkEntry(7,
                       3,
                       zz_coupling=5.9e-4,
                       static_zz_hz=7.8e4,
                       freq_delta_hz=2.0e7),
        CrosstalkEntry(4,
                       10,
                       zz_coupling=2.1e-5,
                       static_zz_hz=3.5e3,
                       freq_delta_hz=8.6e8),
        CrosstalkEntry(9,
                       10,
                       zz_coupling=1.5e-4,
                       static_zz_hz=2.1e4,
                       freq_delta_hz=7.0e7),
        CrosstalkEntry(11,
                       12,
                       zz_coupling=1.4e-4,
                       static_zz_hz=1.9e4,
                       freq_delta_hz=7.0e7),
    ]

    return Target(
        name="transmon_krinner_17q",
        qubits=qubits,
        couplings=couplings,
        crosstalk=crosstalk,
        architecture="transmon",
        attribution=(
            "Krinner et al., 'Realizing Repeated Quantum Error Correction "
            "in a Distance-Three Surface Code', PRX 12, 021049 (2022). "
            "arXiv:2112.03708"),
    )


def transmon_generic(
    n_qubits: int,
    *,
    base_frequency_hz: float = 5.0e9,
    frequency_spread_hz: float = 100e6,
    anharmonicity_hz: float = -330e6,
    coupling_strength_hz: float = 3.0e6,
    t1_us: float = 25.0,
    t2_star_us: float = 15.0,
    topology: str = "linear",
) -> Target:
    """Build a generic transmon target with configurable parameters.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    base_frequency_hz : float
        Frequency of qubit 0.
    frequency_spread_hz : float
        Frequency spacing between adjacent qubits.
    anharmonicity_hz : float
        Uniform anharmonicity (negative for transmon).
    coupling_strength_hz : float
        Uniform nearest-neighbor coupling.
    t1_us, t2_star_us : float
        Uniform decoherence times.
    topology : str
        ``"linear"`` (chain), ``"ring"``, or ``"grid"`` (square lattice
        with side length = ceil(sqrt(n_qubits))).
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

    import math as _math

    qubits = {}
    for i in range(n_qubits):
        qubits[i] = Qubit(
            index=i,
            frequency_hz=base_frequency_hz +
            i * frequency_spread_hz / max(n_qubits - 1, 1),
            anharmonicity_hz=anharmonicity_hz,
            t1_us=t1_us,
            t2_star_us=t2_star_us,
            label=f"Q{i}",
        )

    edges: list[tuple[int, int]] = []
    if topology == "linear":
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
    elif topology == "ring":
        edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    elif topology == "grid":
        side = _math.ceil(_math.sqrt(n_qubits))
        for i in range(n_qubits):
            r, c = divmod(i, side)
            if c + 1 < side and i + 1 < n_qubits:
                edges.append((i, i + 1))
            if r + 1 < side and i + side < n_qubits:
                edges.append((i, i + side))
    else:
        raise ValueError(
            f"Unknown topology {topology!r}. Use 'linear', 'ring', or 'grid'.")

    couplings = [
        Coupling(a, b, coupling_strength_hz=coupling_strength_hz)
        for a, b in edges
    ]

    return Target(
        name=f"transmon_generic_{n_qubits}q_{topology}",
        qubits=qubits,
        couplings=couplings,
        architecture="transmon",
    )
