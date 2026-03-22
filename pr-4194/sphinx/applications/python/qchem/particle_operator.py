import numpy as np
from cudaq import spin


def one_particle_op(p, q):

    if p == q:
        qubit_op_dm = 0.5 * spin.i(p)
        qubit_op_dm -= 0.5 * spin.z(p)

    else:
        coef = 1.0j
        m = -0.25
        if p > q:
            p, q = q, p
            coef = np.conj(coef)

        # Compute the parity string (Z_{p+1}^{q-1})
        z_indices = [i for i in range(p + 1, q)]
        parity_string = 1.0
        for i in z_indices:
            parity_string *= spin.z(i)

        qubit_op_dm = m * spin.x(p) * parity_string * spin.x(q)
        qubit_op_dm += m * spin.y(p) * parity_string * spin.y(q)
        qubit_op_dm -= coef * m * spin.y(p) * parity_string * spin.x(q)
        qubit_op_dm += coef * m * spin.x(p) * parity_string * spin.y(q)

    return qubit_op_dm
