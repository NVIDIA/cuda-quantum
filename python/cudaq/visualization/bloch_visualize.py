# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import matplotlib.pyplot as plt
import numpy as np
from math import isclose  # builtin
from mpl_toolkits.mplot3d import Axes3D
from qutip import Qobj, Bloch
# exposes state class
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


def add_to_bloch_sphere(psi: cudaq_runtime.State,
                        existing_sphere=None,
                        **kwargs) -> Bloch:
    """ 
    Creates a `Bloch` sphere representation of the given single-qubit state. If
    an (optional) existing `Bloch` sphere object is supplied, then adds the
    state to the existing `Bloch` sphere and returns it. The `Bloch` sphere is
    created with QuTiP, and any other keyword arguments provided are passed
    directly to the `qutip.Bloch()` function.  

    Signature:
    ----------
        `add_to_bloch_sphere( psi: cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.State, existing_sphere [Optional]`:
               `None or qutip.Bloch ,**kwargs) -> qutip.Bloch.`
        
    Arguments:
    ----------
        `psi`: A valid single-qubit state, either initialized using the CUDA-Q
               primitives, or via get_state(kernel). 
               A single qubit density matrix is also acceptable.
        `existing_sphere` [Optional]: A `qutip.Bloch` object. If a valid
               `qutip.Bloch` object is not supplied, then creates a new sphere
               with the vector representing the supplied state.
        `kwargs` [Optional]: Optional keyword arguments to be passed to QuTiP
               during `Bloch` sphere initialization. 
        Returns:
        --------
            `Bloch` sphere object. In case existing_sphere is supplied, returns
            a `Bloch` sphere with a copy of its existing data, with added data
            of psi.  
    """

    if not isinstance(psi, cudaq_runtime.State):
        raise TypeError("The supplied argument is not a valid state.")
    if ((not isinstance(existing_sphere, Bloch)) and
        (not existing_sphere == None)):
        existing_sphere = None
        print("Existing sphere object is not a valid Bloch sphere. A new "
              "sphere will be created.")

    b = Bloch(**kwargs) if existing_sphere == None else existing_sphere
    st_rep = np.array(cudaq.StateMemoryView(psi))
    if (st_rep.shape == (2,) and
            isclose(abs(st_rep.dot(st_rep.conjugate())), 1.0,
                    abs_tol=1e-6)) or (st_rep.shape == (2, 2) and
                                       isclose(st_rep.trace().real, 1.0)):
        b.add_states(Qobj(st_rep))
    else:
        raise Exception("The provided argument is not a valid single-qubit "
                        "state or density matrix.")
    return b


def show_bloch_sphere(sphere_data=None, ncols=2, nrows=1) -> None:
    """
    Render the Bloch sphere(s) into a figure. In case a list of Bloch spheres is
    provided, then render all (or some) of the Bloch spheres into a figure
    defined by a grid of `nrows` rows and `ncols` columns.

    Signature:
    ----------
        `show(sphere_data: Bloch object or list thereof , ncols [Optional] = 2, nrows [Optional] = 1)`

    Arguments:
    ----------
        `sphere_data`: An existing `qutip.Bloch` object or list of `qutip.Bloch`
             objects. This is fully interoperational with Bloch sphere objects
             from QuTiP.
        `ncols` [Optional]: Number of columns in the figure, defaults to 2. In
             case sphere_data contains a list of Bloch spheres, this argument
             attempts to fit a maximum of `ncols` columns in the figure.
        `nrows` [Optional]: Number of rows in the figure, defaults to 1.

    Returns:
    --------
        Nothing. Displays the figure in the window. This functionality is
        typically aimed at Jupyter/IPython notebook environments.
    """
    if sphere_data is None:
        print("Nothing to display.")
        return
    else:
        if isinstance(sphere_data, Bloch):
            sphere_data.show()
        elif isinstance(sphere_data, list) and isinstance(
                sphere_data[0], Bloch):
            if (not nrows * ncols >= len(sphere_data)) and nrows > 1:
                raise Exception("Incompatible number of rows and columns for "
                                "sphere_data. Please make sure that nrows*"
                                "ncols={len(sphere_data)} if nrows >=1.")
            fig, axList = plt.subplots(nrows=max(1, nrows),
                                       ncols=min(len(sphere_data), ncols),
                                       subplot_kw={'axes_class': Axes3D})
            if (len(sphere_data) == 1):
                sph = sphere_data[0]
                sph.fig = fig
                sph.axes = axList
                sph.render()
            else:
                for (sph, ax) in zip(sphere_data, axList.flat):
                    sph.fig = fig
                    sph.axes = ax
                    sph.render()

                # make the rest of the axes vanish
                for ax in axList.flat[len(sphere_data):]:
                    ax.grid(False)
                    ax.set_visible(False)

            plt.show()
        else:
            raise TypeError(f"Expected Bloch sphere (or a list of bloch "
                            f"spheres), got {type(sphere_data)}. run the "
                            f"add_to_bloch_sphere() to add a statevector to "
                            f"the bloch sphere before showing it. ")
    return
