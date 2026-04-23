import json
from functools import reduce

try:
    import numpy as np
except ValueError:
    print('numpy should be installed.')

try:
    from pyscf import gto, scf, cc, ao2mo, mp, mcscf, fci, solvent
except ValueError:
    print('PySCF should be installed. Use pip install pyscf')

from pyscf.tools import molden

#######################################################
# Generate the spin molecular orbital hamiltonian


def generate_molecular_spin_ham_restricted(h1e, h2e, ecore):

    # This function generates the molecular spin Hamiltonian
    # H = E_core+sum_{`pq`}  h_{`pq`} a_p^dagger a_q +
    #                          0.5 * h_{`pqrs`} a_p^dagger a_q^dagger a_r a_s
    # h1e: one body integrals h_{`pq`}
    # h2e: two body integrals h_{`pqrs`}
    # `ecore`: constant (nuclear repulsion or core energy in the active space Hamiltonian)

    # Total number of qubits equals the number of spin molecular orbitals
    nqubits = 2 * h1e.shape[0]

    # Initialization
    one_body_coeff = np.zeros((nqubits, nqubits))
    two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))

    ferm_ham = []

    for p in range(nqubits // 2):
        for q in range(nqubits // 2):

            # p & q have the same spin <a|a>= <b|b>=1
            # <a|b>=<b|a>=0 (orthogonal)
            one_body_coeff[2 * p, 2 * q] = h1e[p, q]
            ferm_ham.append(
                f"{one_body_coeff[2 * p, 2 * q]} a_{2 * p}^† a_{2 * q}")

            one_body_coeff[2 * p + 1, 2 * q + 1] = h1e[p, q]
            ferm_ham.append(
                f"{one_body_coeff[2 * p + 1, 2 * q + 1]} b_{2 * p + 1}^† b_{2 * q + 1}"
            )

            for r in range(nqubits // 2):
                for s in range(nqubits // 2):

                    # Same spin (`aaaa`, `bbbbb`) <a|a><a|a>, <b|b><b|b>
                    two_body_coeff[2 * p, 2 * q, 2 * r,
                                   2 * s] = 0.5 * h2e[p, q, r, s]
                    ferm_ham.append(
                        f"{two_body_coeff[2 * p, 2 * q, 2 * r, 2 * s]} "
                        f"a_{2 * p}^† a_{2 * q}^† a_{2 * r} a_{2 * s}")

                    two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                   2 * s + 1] = 0.5 * h2e[p, q, r, s]
                    ferm_ham.append(
                        f"{two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1]} "
                        f"b_{2 * p + 1}^† b_{2 * q + 1}^† b_{2 * r + 1} b_{2 * s + 1}"
                    )

                    # Mixed spin(`abab`, `baba`) <a|a><b|b>, <b|b><a|a>
                    #<a|b>= 0 (orthogonal)
                    two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                   2 * s] = 0.5 * h2e[p, q, r, s]
                    ferm_ham.append(
                        f"{two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1, 2 * s]} "
                        f"a_{2 * p}^† b_{2 * q + 1}^† b_{2 * r + 1} a_{2 * s}")

                    two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                   2 * s + 1] = 0.5 * h2e[p, q, r, s]
                    ferm_ham.append(
                        f"{two_body_coeff[2 * p + 1, 2 * q, 2 * r, 2 * s + 1]} "
                        f"b_{2 * p + 1}^† a_{2 * q}^† a_{2 * r} b_{2 * s + 1}")

    full_hamiltonian = " + ".join(ferm_ham)

    return one_body_coeff, two_body_coeff, ecore, full_hamiltonian


def generate_pe_spin_ham_restricted(v_pe):

    # Total number of qubits equals the number of spin molecular orbitals
    nqubits = 2 * v_pe.shape[0]

    # Initialization
    spin_pe_op = np.zeros((nqubits, nqubits))

    for p in range(nqubits // 2):
        for q in range(nqubits // 2):

            # p & q have the same spin <a|a>= <b|b>=1
            # <a|b>=<b|a>=0 (orthogonal)
            spin_pe_op[2 * p, 2 * q] = v_pe[p, q]
            spin_pe_op[2 * p + 1, 2 * q + 1] = v_pe[p, q]

    return spin_pe_op


def a_idx(p):
    return 2 * p  # alpha spin-orbital index


def b_idx(p):
    return 2 * p + 1  # beta  spin-orbital index


def generate_molecular_spin_ham_ur(
        h1e_alpha,
        h1e_beta,
        h2e_alpha_alpha,  # `(pq|rs)`
        h2e_beta_beta,  # `(pq|rs)`
        h2e_alpha_beta,  # `(pq|rs)`
        h2e_beta_alpha,  # `(pq|rs)`
        ecore):
    """
    Build the UHF spin-orbital Hamiltonian in chemists' notation:

        H = e_core
          `+ sum_{pq,σ} h_{pq}^σ a†_{pσ} a_{qσ}`
          `+ 1/2 sum_{pqrs} sum_{σ,τ} (pq|rs) a†_{pσ} a†_{qτ} a_{sτ} a_{rσ}`

    Inputs:
      `h1e_alpha, h1e_beta:  (nmo, nmo) one-electron MO integrals for α and β`
      `h2e_*:                (nmo, nmo, nmo, nmo) two-electron MO integrals (pq|rs)`
      `ecore:                nuclear repulsion`

    Returns:
      `one_body_coeff: (2nmo, 2nmo)`
      `two_body_coeff: (2nmo, 2nmo, 2nmo, 2nmo)`
      `ecore:          float`
      `full_hamiltonian: string with fermionic operators (for inspection)`
    """
    n_mos = h1e_alpha.shape[0]
    nqubits = 2 * n_mos

    one_body_coeff = np.zeros((nqubits, nqubits), dtype=np.complex128)
    two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits),
                              dtype=np.complex128)
    ferm_ham = []

    # ---------- One-body terms ----------
    for p in range(n_mos):
        for q in range(n_mos):
            # Alpha spin
            val_a = h1e_alpha[p, q]
            one_body_coeff[a_idx(p), a_idx(q)] = val_a
            ferm_ham.append(f"{val_a} a_{a_idx(p)}^† a_{a_idx(q)}")

            # Beta spin
            val_b = h1e_beta[p, q]
            one_body_coeff[b_idx(p), b_idx(q)] = val_b
            ferm_ham.append(f"{val_b} b_{b_idx(p)}^† b_{b_idx(q)}")

    # ---------- Two-body terms ----------
    half = 0.5
    for p in range(n_mos):
        for q in range(n_mos):
            for r in range(n_mos):
                for s in range(n_mos):
                    # --- αα ---
                    val = half * h2e_alpha_alpha[p, q, r, s]
                    two_body_coeff[a_idx(p),
                                   a_idx(q),
                                   a_idx(s),
                                   a_idx(r)] += val
                    ferm_ham.append(
                        f"{val} a_{a_idx(p)}^† a_{a_idx(q)}^† a_{a_idx(s)} a_{a_idx(r)}"
                    )

                    # --- ββ ---
                    val = half * h2e_beta_beta[p, q, r, s]
                    two_body_coeff[b_idx(p),
                                   b_idx(q),
                                   b_idx(s),
                                   b_idx(r)] += val
                    ferm_ham.append(
                        f"{val} b_{b_idx(p)}^† b_{b_idx(q)}^† b_{b_idx(s)} b_{b_idx(r)}"
                    )

                    # --- αβ ---
                    # a_{pα}† b_{qβ}† b_{sβ} a_{rα}
                    val = half * h2e_alpha_beta[p, q, r, s]
                    two_body_coeff[a_idx(p),
                                   b_idx(q),
                                   b_idx(s),
                                   a_idx(r)] += val
                    ferm_ham.append(
                        f"{val} a_{a_idx(p)}^† b_{b_idx(q)}^† b_{b_idx(s)} a_{a_idx(r)}"
                    )

                    # --- βα ---
                    # b_{pβ}† a_{qα}† a_{sα} b_{rβ}
                    val = half * h2e_beta_alpha[p, q, r, s]
                    two_body_coeff[b_idx(p),
                                   a_idx(q),
                                   a_idx(s),
                                   b_idx(r)] += val
                    ferm_ham.append(
                        f"{val} b_{b_idx(p)}^† a_{a_idx(q)}^† a_{a_idx(s)} b_{b_idx(r)}"
                    )

    full_hamiltonian = " + ".join(ferm_ham)
    return one_body_coeff, two_body_coeff, ecore, full_hamiltonian


def create_energy_dict():
    """
    Create a dictionary to store energy values from different calculation methods.
    
    Args:
        calculation_type (`str`): 'gas' for gas phase or '`pe`' for `polarizable` embedding calculations
    
    Returns:
        dict: Dictionary with energy tracking structure
    """
    energy_dict = {
        'hf': None,  # Hartree-Fock energy
        'mp2': None,  # MP2 energy
        'casci': None,  # CASCI energy
        'casscf': None,  # CASSCF energy
        'ccsd': None,  # CCSD energy
        'fci': None,  # FCI energy
        'nuclear_repulsion': None  # Nuclear repulsion energy
    }

    return energy_dict


####################################################################

# A- Gas phase simulation
#############################
## Beginning of simulation
#############################

def get_mol_hamiltonian(xyz:str, spin:int, charge: int, basis:str, symmetry:bool = False, memory:float = 4000, cycles:int = 100, \
                              initguess:str = 'minao', UR:bool = False, nele_cas = None, norb_cas = None, MP2:bool = False, natorb:bool = False,\
                              casci:bool = False, ccsd:bool = False, casscf:bool = False, integrals_natorb:bool = False, \
                                integrals_casscf:bool = False, viz_orb:bool = False, verbose:bool = False):

    # Initialize energy dictionary
    energies = create_energy_dict()
    ################################
    # Initialize the molecule
    ################################
    filename = xyz.split('.')[0]

    if (nele_cas is None) and (norb_cas is not None):
        raise ValueError(
            "WARN: nele_cas is None and norb_cas is not None. "
            "nele_cas and norb_cas should be either both None or have values")

    if (nele_cas is not None) and (norb_cas is None):
        raise ValueError(
            "WARN: nele_cas is not None and norb_cas is None. "
            "nele_cas and norb_cas should be either both None or have values")

    #if UR and `nele_cas` is None:
    #    raise ValueError("WARN: Unrestricted spin calculation for the full space is not supported yet on Cudaq.\
    #                  Only active space is currently supported for the unrestricted spin calculations.")

    if UR and integrals_natorb:
        print(
            "[pyscf] WARNING: Natural orbitals are not supported for unrestricted calculations. "
            "Setting integrals_natorb to False. HF molecular orbitals will be used for calculating integrals."
        )
        integrals_natorb = False

    if UR and integrals_casscf:
        print(
            "[pyscf] WARNING: CASSCF integrals are not supported for unrestricted calculations. "
            "Setting integrals_casscf to False. HF molecular orbitals will be used for calculating integrals."
        )
        integrals_casscf = False

    ########################################################################
    # To add (coming soon)

    mol = gto.M(atom=xyz,
                spin=spin,
                charge=charge,
                basis=basis,
                max_memory=memory,
                symmetry=symmetry,
                output=filename + '-pyscf.log',
                verbose=4)

    ##################################
    # Mean field (HF)
    ##################################
    if UR:
        myhf = scf.UHF(mol)
        myhf.max_cycle = cycles
        myhf.chkfile = filename + '-pyscf.chk'
        myhf.init_guess = initguess
        myhf.kernel()

        norb = myhf.mo_coeff[0].shape[1]
        if verbose:
            print('[pyscf] Total number of alpha molecular orbitals = ', norb)
        norb = myhf.mo_coeff[1].shape[1]
        if verbose:
            print('[pyscf] Total number of beta molecular orbitals = ', norb)

    else:

        if spin == 0:
            myhf = scf.RHF(mol)
            myhf.max_cycle = cycles
            myhf.chkfile = filename + '-pyscf.chk'
            myhf.init_guess = initguess
            myhf.kernel()

            norb = myhf.mo_coeff.shape[1]
            if verbose:
                print('[pyscf] Total number of orbitals = ', norb)
        else:
            myhf = scf.ROHF(mol)
            myhf.max_cycle = cycles
            myhf.chkfile = filename + '-pyscf.chk'
            myhf.init_guess = initguess
            myhf.kernel()

            norb = myhf.mo_coeff.shape[1]
            if verbose:
                print('[pyscf] Total number of orbitals = ', norb)

    nelec = mol.nelectron
    if verbose:
        print('[pyscf] Total number of electrons = ', nelec)
    if verbose:
        print('[pyscf] HF energy = ', myhf.e_tot)
    if viz_orb:
        molden.from_mo(mol, filename + '_HF_molorb.molden', myhf.mo_coeff)

    energies['hf'] = myhf.e_tot
    energies['nuclear_repulsion'] = myhf.energy_nuc()

    if not myhf.converged:

        raise ValueError("[pyscf] WARNING: HF calculation did not converge!")

    ##########################
    # MP2
    ##########################
    if MP2:

        if UR:
            mymp = mp.UMP2(myhf)
            mp_ecorr, mp_t2 = mymp.kernel()
            if verbose:
                print('[pyscf] UR-MP2 energy= ', mymp.e_tot)

            if integrals_natorb or natorb:
                # Compute natural orbitals
                dma, dmb = mymp.make_rdm1()
                noon_a, U_a = np.linalg.eigh(dma)
                noon_b, U_b = np.linalg.eigh(dmb)
                noon_a = np.flip(noon_a)
                noon_b = np.flip(noon_b)

                if verbose:
                    print(
                        '[pyscf] Natural orbital (alpha orbitals) occupation number from UR-MP2: '
                    )
                if verbose:
                    print(noon_a)
                if verbose:
                    print(
                        '[pyscf] Natural orbital (beta orbitals) occupation number from UR-MP2: '
                    )
                if verbose:
                    print(noon_b)

                natorbs = np.zeros(np.shape(myhf.mo_coeff))
                natorbs[0, :, :] = np.dot(myhf.mo_coeff[0], U_a)
                natorbs[0, :, :] = np.fliplr(natorbs[0, :, :])
                natorbs[1, :, :] = np.dot(myhf.mo_coeff[1], U_b)
                natorbs[1, :, :] = np.fliplr(natorbs[1, :, :])

        else:
            if spin != 0:
                raise ValueError("WARN: ROMP2 is unvailable in pyscf.")
            else:
                mymp = mp.MP2(myhf)
                mp_ecorr, mp_t2 = mymp.kernel()
                if verbose:
                    print('[pyscf] R-MP2 energy= ', mymp.e_tot)

                if integrals_natorb or natorb:
                    # Compute natural orbitals
                    noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
                    if verbose:
                        print(
                            '[pyscf] Natural orbital occupation number from R-MP2: '
                        )
                    if verbose:
                        print(noons)
                    if viz_orb:
                        molden.from_mo(mol, filename + '_MP2_natorb.molden',
                                       natorbs)

        energies['mp2'] = mymp.e_tot
    #######################################
    # CASCI if active space is defined
    # FCI if the active space is None
    ######################################
    if casci:

        if UR:

            if nele_cas is None:
                myfci = fci.FCI(myhf)
                result = myfci.kernel()
                energies['fci'] = result[0]
                if verbose:
                    print('[pyscf] FCI energy = ', result[0])

            else:
                # Convert `nele_cas` to the correct format for UCASCI
                if isinstance(nele_cas, (int, float)):
                    # For unrestricted calculations, `nele_cas` must be a tuple (alpha, beta)
                    nelec_beta = (nele_cas - mol.spin) // 2
                    nelec_alpha = nele_cas - nelec_beta
                    nele_cas_tuple = (nelec_alpha, nelec_beta)
                    if verbose:
                        print(
                            f'[pyscf] Converting nele_cas from {nele_cas} to {nele_cas_tuple} (alpha, beta)'
                        )
                else:
                    # If already a tuple, use as is
                    nele_cas_tuple = nele_cas

                if natorb:

                    mycasci = mcscf.UCASCI(myhf, norb_cas, nele_cas_tuple)
                    mycasci.kernel(natorbs)
                    if verbose:
                        print(
                            '[pyscf] UR-CASCI energy using natural orbitals= ',
                            mycasci.e_tot)
                else:

                    mycasci = mcscf.UCASCI(myhf, norb_cas, nele_cas_tuple)
                    mycasci.kernel()
                    if verbose:
                        print(
                            '[pyscf] UR-CASCI energy using molecular orbitals= ',
                            mycasci.e_tot)
                energies['casci'] = mycasci.e_tot

        else:

            if nele_cas is None:
                myfci = fci.FCI(myhf)
                result = myfci.kernel()
                energies['fci'] = result[0]
                if verbose:
                    print('[pyscf] FCI energy = ', result[0])

            else:
                if natorb and (spin == 0):
                    mycasci = mcscf.CASCI(myhf, norb_cas, nele_cas)
                    mycasci.kernel(natorbs)
                    if verbose:
                        print('[pyscf] R-CASCI energy using natural orbitals= ',
                              mycasci.e_tot)

                elif natorb and (spin != 0):
                    raise ValueError(
                        "WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf."
                    )

                else:
                    mycasci = mcscf.CASCI(myhf, norb_cas, nele_cas)
                    mycasci.kernel()
                    if verbose:
                        print(
                            '[pyscf] R-CASCI energy using molecular orbitals= ',
                            mycasci.e_tot)

                energies['casci'] = mycasci.e_tot

    ########################
    # CCSD
    ########################
    if ccsd:

        if UR:
            if nele_cas is None:
                mycc = myhf.CCSD().run()
                if verbose:
                    print('[pyscf] Total UR-CCSD energy = ', mycc.e_tot)

            else:

                if isinstance(nele_cas, (int, float)):
                    # For unrestricted calculations, `nele_cas` must be a tuple (alpha, beta)
                    nelec_beta = (nele_cas - mol.spin) // 2
                    nelec_alpha = nele_cas - nelec_beta
                    nele_cas_tuple = (nelec_alpha, nelec_beta)
                    if verbose:
                        print(
                            f'[pyscf] Converting nele_cas from {nele_cas} to {nele_cas_tuple} (alpha, beta)'
                        )
                else:
                    # If already a tuple, use as is
                    nele_cas_tuple = nele_cas

                mc = mcscf.UCASCI(myhf, norb_cas, nele_cas_tuple)
                frozen = []
                frozen = [y for y in range(0, mc.ncore[0])]
                frozen += [
                    y for y in range(mc.ncore[0] +
                                     mc.ncas, len(myhf.mo_coeff[0]))
                ]

                if natorb:
                    mycc = cc.UCCSD(myhf, frozen=frozen, mo_coeff=natorbs)
                    mycc.max_cycle = cycles
                    mycc.kernel()
                    if verbose:
                        print(
                            '[pyscf] UR-CCSD energy of the active space using natural orbitals= ',
                            mycc.e_tot)

                else:
                    mycc = cc.UCCSD(myhf, frozen=frozen)
                    mycc.max_cycle = cycles
                    mycc.kernel()
                    if verbose:
                        print(
                            '[pyscf] UR-CCSD energy of the active space using molecular orbitals= ',
                            mycc.e_tot)

            energies['ccsd'] = mycc.e_tot

        else:
            if nele_cas is None:
                mycc = cc.CCSD(myhf)
                mycc.max_cycle = cycles
                mycc.kernel()
                if verbose:
                    print('[pyscf] Total R-CCSD energy = ', mycc.e_tot)

            else:
                mc = mcscf.CASCI(myhf, norb_cas, nele_cas)
                frozen = []
                frozen += [y for y in range(0, mc.ncore)]
                frozen += [
                    y for y in range(mc.ncore + norb_cas, len(myhf.mo_coeff))
                ]
                if natorb and (spin == 0):
                    mycc = cc.CCSD(myhf, frozen=frozen, mo_coeff=natorbs)
                    mycc.max_cycle = cycles
                    mycc.kernel()
                    if verbose:
                        print(
                            '[pyscf] R-CCSD energy of the active space using natural orbitals= ',
                            mycc.e_tot)

                elif natorb and (spin != 0):
                    raise ValueError(
                        "WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf."
                    )

                else:
                    mycc = cc.CCSD(myhf, frozen=frozen)
                    mycc.max_cycle = cycles
                    mycc.kernel()
                    if verbose:
                        print(
                            '[pyscf] R-CCSD energy of the active space using molecular orbitals= ',
                            mycc.e_tot)

            energies['ccsd'] = mycc.e_tot

    #########################
    # CASSCF
    #########################
    if casscf:

        if nele_cas is None:
            raise ValueError("WARN: You should define the active space.")

        if UR:
            if isinstance(nele_cas, (int, float)):
                # For unrestricted calculations, `nele_cas` must be a tuple (alpha, beta)
                nelec_beta = (nele_cas - mol.spin) // 2
                nelec_alpha = nele_cas - nelec_beta
                nele_cas_tuple = (nelec_alpha, nelec_beta)
                if verbose:
                    print(
                        f'[pyscf] Converting nele_cas from {nele_cas} to {nele_cas_tuple} (alpha, beta)'
                    )
            else:
                # If already a tuple, use as is
                nele_cas_tuple = nele_cas

            if natorb:
                mycas = mcscf.UCASSCF(myhf, norb_cas, nele_cas_tuple)
                mycas.max_cycle_macro = cycles
                mycas.kernel(natorbs)
                if verbose:
                    print('[pyscf] UR-CASSCF energy using natural orbitals= ',
                          mycas.e_tot)
            else:
                mycas = mcscf.UCASSCF(myhf, norb_cas, nele_cas_tuple)
                mycas.max_cycle_macro = cycles
                mycas.kernel()
                if verbose:
                    print('[pyscf] UR-CASSCF energy using molecular orbitals= ',
                          mycas.e_tot)

            energies['casscf'] = mycas.e_tot
        else:

            if natorb and (spin == 0):
                mycas = mcscf.CASSCF(myhf, norb_cas, nele_cas)
                mycas.max_cycle_macro = cycles
                mycas.kernel(natorbs)
                if verbose:
                    print('[pyscf] R-CASSCF energy using natural orbitals= ',
                          mycas.e_tot)

            elif natorb and (spin != 0):
                raise ValueError(
                    "WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf."
                )

            else:
                mycas = mcscf.CASSCF(myhf, norb_cas, nele_cas)
                mycas.max_cycle_macro = cycles
                mycas.kernel()
                if verbose:
                    print('[pyscf] R-CASSCF energy using molecular orbitals= ',
                          mycas.e_tot)

        energies['casscf'] = mycas.e_tot

    ###################################
    # CASCI: `FCI` of the active space
    ##################################
    if casci and casscf:

        if UR:
            h1e_cas, ecore = mycas.get_h1eff()
            h2e_cas = mycas.get_h2eff()

            e_fci, fcivec = fci.direct_uhf.kernel(h1e_cas,
                                                  h2e_cas,
                                                  norb_cas,
                                                  nele_cas_tuple,
                                                  ecore=ecore)
            if verbose:
                print('[pyscf] UR-CASCI energy using the casscf orbitals= ',
                      e_fci)

        else:
            if natorb and (spin != 0):
                raise ValueError(
                    "WARN: Natural orbitals cannot be computed. ROMP2 is unavailable in pyscf."
                )
            else:
                h1e_cas, ecore = mycas.get_h1eff()
                h2e_cas = mycas.get_h2eff()

                e_fci, fcivec = fci.direct_spin1.kernel(h1e_cas,
                                                        h2e_cas,
                                                        norb_cas,
                                                        nele_cas,
                                                        ecore=ecore)
                if verbose:
                    print('[pyscf] R-CASCI energy using the casscf orbitals= ',
                          e_fci)

    ###################################################################################
    # Computation of one- and two- electron integrals for the active space Hamiltonian
    ###################################################################################

    if UR:
        if nele_cas is None:
            # Get one-electron integrals in AO basis
            h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

            # Convert one-electron integrals to MO basis for alpha and beta spins
            h1e_alpha = reduce(np.dot,
                               (myhf.mo_coeff[0].T, h1e_ao, myhf.mo_coeff[0]))
            h1e_beta = reduce(np.dot,
                              (myhf.mo_coeff[1].T, h1e_ao, myhf.mo_coeff[1]))

            # Get two-electron integrals in AO basis
            h2e_ao = mol.intor(
                "int2e_sph",
                aosym='s1')  # Use 's1' for unrestricted calculations
            nmo = myhf.mo_coeff[0].shape[1]

            # Convert two-electron integrals to MO basis
            eri_aa = ao2mo.incore.general(h2e_ao,
                                          (myhf.mo_coeff[0], myhf.mo_coeff[0],
                                           myhf.mo_coeff[0], myhf.mo_coeff[0]),
                                          compact=False).reshape(
                                              nmo, nmo, nmo, nmo)
            eri_bb = ao2mo.incore.general(h2e_ao,
                                          (myhf.mo_coeff[1], myhf.mo_coeff[1],
                                           myhf.mo_coeff[1], myhf.mo_coeff[1]),
                                          compact=False).reshape(
                                              nmo, nmo, nmo, nmo)
            eri_ab = ao2mo.incore.general(h2e_ao,
                                          (myhf.mo_coeff[0], myhf.mo_coeff[0],
                                           myhf.mo_coeff[1], myhf.mo_coeff[1]),
                                          compact=False).reshape(
                                              nmo, nmo, nmo, nmo)
            eri_ba = ao2mo.incore.general(h2e_ao,
                                          (myhf.mo_coeff[1], myhf.mo_coeff[1],
                                           myhf.mo_coeff[0], myhf.mo_coeff[0]),
                                          compact=False).reshape(
                                              nmo, nmo, nmo, nmo)

            # Reorder integrals from `(pr|qs) to (pq|rs)`
            h2e_alpha_alpha = eri_aa.transpose(0, 2, 1, 3)
            h2e_beta_beta = eri_bb.transpose(0, 2, 1, 3)
            h2e_alpha_beta = eri_ab.transpose(0, 2, 1, 3)
            h2e_beta_alpha = eri_ba.transpose(0, 2, 1, 3)

            nuclear_repulsion = myhf.energy_nuc()

            obi, tbi, e_nn, ferm_ham = generate_molecular_spin_ham_ur(
                h1e_alpha, h1e_beta, h2e_alpha_alpha, h2e_beta_beta,
                h2e_alpha_beta, h2e_beta_alpha, nuclear_repulsion)

        else:

            if isinstance(nele_cas, (int, float)):
                # For unrestricted calculations, `nele_cas` must be a tuple (alpha, beta)
                nelec_beta = (nele_cas - mol.spin) // 2
                nelec_alpha = nele_cas - nelec_beta

                nele_cas_tuple = (nelec_alpha, nelec_beta)
                if verbose:
                    print(
                        f'[pyscf] Converting nele_cas from {nele_cas} to {nele_cas_tuple} (alpha, beta)'
                    )
            else:
                # If already a tuple, use as is
                nele_cas_tuple = nele_cas

            mc = mcscf.UCASCI(myhf, norb_cas, nele_cas_tuple)
            (h1e_alpha_cas, h1e_beta_cas), ecore = mc.get_h1eff(myhf.mo_coeff)

            # 1. Get the density matrices for the frozen core orbitals
            # The number of core orbitals for alpha and beta spins
            ncore_a = myhf.nelec[0] - nele_cas_tuple[0]
            ncore_b = myhf.nelec[1] - nele_cas_tuple[1]

            # Get two-electron integrals in AO basis
            h2e_ao = mol.intor("int2e_sph", aosym='s1')

            # Select the active space MOs and transform the integrals
            # The number of core orbitals
            active_idx_a = slice(ncore_a, ncore_a + norb_cas)
            active_idx_b = slice(ncore_b, ncore_b + norb_cas)
            active_mo_a = myhf.mo_coeff[0][:, active_idx_a]
            active_mo_b = myhf.mo_coeff[1][:, active_idx_b]

            # Transform 2-e integrals to the active space MO basis
            eri_aa = ao2mo.incore.general(
                h2e_ao, (active_mo_a, active_mo_a, active_mo_a, active_mo_a),
                compact=False).reshape(norb_cas, norb_cas, norb_cas, norb_cas)
            eri_bb = ao2mo.incore.general(
                h2e_ao, (active_mo_b, active_mo_b, active_mo_b, active_mo_b),
                compact=False).reshape(norb_cas, norb_cas, norb_cas, norb_cas)
            eri_ab = ao2mo.incore.general(
                h2e_ao, (active_mo_a, active_mo_a, active_mo_b, active_mo_b),
                compact=False).reshape(norb_cas, norb_cas, norb_cas, norb_cas)
            eri_ba = ao2mo.incore.general(
                h2e_ao, (active_mo_b, active_mo_b, active_mo_a, active_mo_a),
                compact=False).reshape(norb_cas, norb_cas, norb_cas, norb_cas)

            # 7. Reorder integrals from `(pr|qs) to (pq|rs)`
            h2e_alpha_alpha = eri_aa.transpose(0, 2, 1, 3)
            h2e_beta_beta = eri_bb.transpose(0, 2, 1, 3)
            h2e_alpha_beta = eri_ab.transpose(0, 2, 1, 3)
            h2e_beta_alpha = eri_ba.transpose(0, 2, 1, 3)

            # Compute the molecular spin electronic Hamiltonian from the
            # molecular electron integrals

            obi, tbi, core_energy, ferm_ham = generate_molecular_spin_ham_ur(
                h1e_alpha_cas, h1e_beta_cas, h2e_alpha_alpha, h2e_beta_beta,
                h2e_alpha_beta, h2e_beta_alpha, ecore)

    else:

        if nele_cas is None:
            # Compute the 1e integral in atomic orbital then convert to HF basis
            h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            ## Ways to convert from `ao` to mo
            #h1e=`np.einsum('pi,pq,qj->ij', myhf.mo_coeff, h1e_ao, myhf.mo_coeff)`
            h1e = reduce(np.dot, (myhf.mo_coeff.T, h1e_ao, myhf.mo_coeff))
            #h1e=`reduce(np.dot, (myhf.mo_coeff.conj().T, h1e_ao, myhf.mo_coeff))`

            # Compute the 2e integrals then convert to HF basis
            h2e_ao = mol.intor("int2e_sph", aosym='1')
            h2e = ao2mo.incore.full(h2e_ao, myhf.mo_coeff)

            # `Reorder the chemist notation (pq|rs) ERI h_prqs to h_pqrs`
            # a_p^dagger a_r a_q^dagger a_s --> a_p^dagger a_q^dagger a_r a_s
            h2e = h2e.transpose(0, 2, 3, 1)

            nuclear_repulsion = myhf.energy_nuc()

            # Compute the molecular spin electronic Hamiltonian from the
            # molecular electron integrals
            obi, tbi, e_nn, ferm_ham = generate_molecular_spin_ham_restricted(
                h1e, h2e, nuclear_repulsion)

        else:

            if integrals_natorb:
                if spin != 0:
                    raise ValueError("WARN: ROMP2 is unvailable in pyscf.")
                else:
                    mc = mcscf.CASCI(myhf, norb_cas, nele_cas)
                    h1e_cas, ecore = mc.get_h1eff(natorbs)
                    h2e_cas = mc.get_h2eff(natorbs)
                    h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                    h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1),
                                         order='C')

            elif integrals_casscf:
                if casscf:
                    h1e_cas, ecore = mycas.get_h1eff()
                    h2e_cas = mycas.get_h2eff()
                    h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                    h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1),
                                         order='C')
                else:
                    raise ValueError(
                        "WARN: You need to run casscf. Use casscf=True.")

            else:
                mc = mcscf.CASCI(myhf, norb_cas, nele_cas)
                h1e_cas, ecore = mc.get_h1eff(myhf.mo_coeff)
                h2e_cas = mc.get_h2eff(myhf.mo_coeff)
                h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')

            # Compute the molecular spin electronic Hamiltonian from the
            # molecular electron integrals
            obi, tbi, core_energy, ferm_ham = generate_molecular_spin_ham_restricted(
                h1e_cas, h2e_cas, ecore)

        # Dump obi and `tbi` to binary file.
        # `obi.astype(complex).tofile(f'{filename}_one_body.dat')`
        # `tbi.astype(complex).tofile(f'{filename}_two_body.dat')`

    ######################################################
    # Dump energies / etc to a metadata file
    if nele_cas is None:
        # `metadata = {'num_electrons': nelec, 'num_orbitals': norb, 'nuclear_energy': e_nn, 'hf_energy': myhf.e_tot}`
        # with open(f'{filename}_metadata.`json`', 'w') as f:
        #        `json`.dump(metadata, f)
        return (obi, tbi, e_nn, nelec, norb, ferm_ham, energies)

    else:
        # `metadata = {'num_electrons_cas': nele_cas, 'num_orbitals_cas': norb_cas, 'core_energy': ecore, 'hf_energy': myhf.e_tot}`
        # with open(f'{filename}_metadata.`json`', 'w') as f:
        #        `json`.dump(metadata, f)
        return (obi, tbi, ecore, nele_cas, norb_cas, ferm_ham, energies)


#######################################################
# B- With `polarizable` embedded framework

def get_mol_pe_hamiltonian(xyz:str, potfile:str, spin:int, charge: int, basis:str, symmetry:bool=False, memory:float=4000, cycles:int=100,\
                       initguess:str='minao', nele_cas=None, norb_cas=None, MP2:bool=False, natorb:bool=False, casci:bool=False, \
                        ccsd:bool=False, casscf:bool=False, integrals_natorb:bool=False, integrals_casscf:bool=False, verbose:bool=False):

    from qchem.cppe_lib import PolEmbed

    # Initialize energy dictionary
    energies = create_energy_dict()

    if spin != 0:
        print(
            'WARN: UHF is not implemented yet for PE model. RHF & ROHF are only supported.'
        )

    ################################
    # Initialize the molecule
    ################################
    filename = xyz.split('.')[0]
    mol = gto.M(atom=xyz,
                spin=spin,
                charge=charge,
                basis=basis,
                max_memory=memory,
                symmetry=symmetry,
                output=filename + '-pyscf.log',
                verbose=4)

    ##################################
    # Mean field (HF)
    ##################################

    nelec = mol.nelectron
    if verbose:
        print('[Pyscf] Total number of electrons = ', nelec)

    # HF with PE model.
    mf_pe = scf.RHF(mol)
    mf_pe.init_guess = initguess
    mf_pe.chkfile = filename + '-pyscf.chk'
    mf_pe = solvent.PE(mf_pe, potfile).run()
    norb = mf_pe.mo_coeff.shape[1]
    if verbose:
        print('[Pyscf] Total number of orbitals = ', norb)
    if verbose:
        print('[Pyscf] Total HF energy with solvent:', mf_pe.e_tot)
    if verbose:
        print('[Pyscf] Polarizable embedding energy from HF: ',
              mf_pe.with_solvent.e)

    dm = mf_pe.make_rdm1()

    energies['hf'] = mf_pe.e_tot

    ##################
    # MP2
    ##################
    if MP2:

        if spin != 0:
            raise ValueError("WARN: ROMP2 is unvailable in pyscf.")
        else:
            mymp = mp.MP2(mf_pe)
            mymp = solvent.PE(mymp, potfile)
            mymp.run()
            if verbose:
                print('[pyscf] R-MP2 energy with solvent= ', mymp.e_tot)
            if verbose:
                print('[Pyscf] Polarizable embedding energy from MP: ',
                      mymp.with_solvent.e)

            if integrals_natorb or natorb:
                # Compute natural orbitals
                noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
                if verbose:
                    print(
                        '[Pyscf] Natural orbital occupation number from R-MP2: '
                    )
                if verbose:
                    print(noons)

        energies['mp2'] = mymp.e_tot

    #################
    # CASCI
    #################
    if casci:

        if nele_cas is None:

            #`myfci`=`fci`.`FCI`(`mf`_`pe`)
            #`myfci`=solvent.PE(`myfci`, `args`.`potfile`, `dm`)
            #`myfci`.run()
            #if verbose: print('[`pyscf`] FCI energy with solvent= ', `myfci`.e_tot)
            #if verbose: print('[`Pyscf`] Polarizable embedding energy from FCI: ', `myfci`.with_solvent.e)
            print('[Pyscf] FCI with PE is not supported.')

        else:
            if natorb and (spin == 0):
                mycasci = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
                mycasci = solvent.PE(mycasci, potfile)
                mycasci.run(natorbs)
                if verbose:
                    print(
                        '[pyscf] CASCI energy (using natural orbitals) with solvent= ',
                        mycasci.e_tot)

            else:
                mycasci = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
                mycasci = solvent.PE(mycasci, potfile)
                mycasci.run()
                if verbose:
                    print(
                        '[pyscf] CASCI energy (using molecular orbitals) with solvent= ',
                        mycasci.e_tot)
                if verbose:
                    print('[Pyscf] Polarizable embedding energy from CASCI: ',
                          mycasci.with_solvent.e)
        energies['casci'] = mycasci.e_tot

    #################
    ## CCSD
    #################
    if ccsd:

        if nele_cas is None:
            mycc = cc.CCSD(mf_pe)
            mycc = solvent.PE(mycc, potfile)
            mycc.run()
            if verbose:
                print('[Pyscf] Total CCSD energy with solvent: ', mycc.e_tot)
            if verbose:
                print('[Pyscf] Polarizable embedding energy from CCSD: ',
                      mycc.with_solvent.e)

        else:
            mc = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
            frozen = []
            frozen += [y for y in range(0, mc.ncore)]
            frozen += [
                y for y in range(mc.ncore + norb_cas, len(mf_pe.mo_coeff))
            ]

            if natorb and (spin == 0):
                mycc = cc.CCSD(mf_pe, frozen=frozen, mo_coeff=natorbs)
                mycc = solvent.PE(mycc, potfile)
                mycc.run()
                if verbose:
                    print(
                        '[pyscf] R-CCSD energy of the active space (using natural orbitals) with solvent= ',
                        mycc.e_tot)
            else:
                mycc = cc.CCSD(mf_pe, frozen=frozen)
                mycc = solvent.PE(mycc, potfile)
                mycc.run()
                if verbose:
                    print(
                        '[pyscf] CCSD energy of the active space (using molecular orbitals) with solvent= ',
                        mycc.e_tot)
                if verbose:
                    print('[Pyscf] Polarizable embedding energy from CCSD: ',
                          mycc.with_solvent.e)

        energies['ccsd'] = mycc.e_tot
    ############################
    # CASSCF
    ############################
    if casscf:
        if natorb and (spin == 0):
            mycas = mcscf.CASSCF(mf_pe, norb_cas, nele_cas)
            mycas = solvent.PE(mycas, potfile)
            mycas.max_cycle_macro = cycles
            mycas.kernel(natorbs)
            if verbose:
                print(
                    '[pyscf] CASSCF energy (using natural orbitals) with solvent= ',
                    mycas.e_tot)

        else:
            mycas = mcscf.CASSCF(mf_pe, norb_cas, nele_cas)
            mycas = solvent.PE(mycas, potfile)
            mycas.max_cycle_macro = cycles
            mycas.kernel()
            if verbose:
                print(
                    '[pyscf] CASSCF energy (using molecular orbitals) with solvent= ',
                    mycas.e_tot)

        energies['casscf'] = mycas.e_tot

    ###########################################################################
    # Computation of one and two electron integrals for the QC+PE
    ###########################################################################
    if nele_cas is None:
        # Compute the 1e integral in atomic orbital then convert to HF basis
        h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        ## Ways to convert from `ao` to mo
        #h1e=`np`.`einsum`('pi,`pq`,`qj`->`ij`', `myhf`.mo_`coeff`, h1e_`ao`, `myhf`.mo_`coeff`)
        h1e = reduce(np.dot, (mf_pe.mo_coeff.T, h1e_ao, mf_pe.mo_coeff))
        #h1e=reduce(`np`.dot, (`myhf`.mo_`coeff`.conj().T, h1e_`ao`, `myhf`.mo_`coeff`))

        # Compute the 2e integrals then convert to HF basis
        h2e_ao = mol.intor("int2e_sph", aosym='1')
        h2e = ao2mo.incore.full(h2e_ao, mf_pe.mo_coeff)

        # Reorder the chemist notation (`pq`|rs) ERI h_`prqs` to h_`pqrs`
        # a_p^dagger a_r a_q^dagger a_s --> a_p^dagger a_q^dagger a_r a_s
        h2e = h2e.transpose(0, 2, 3, 1)

        nuclear_repulsion = mf_pe.energy_nuc()

        # Compute the molecular spin electronic Hamiltonian from the
        # molecular electron integrals
        obi, tbi, e_nn, ferm_ham = generate_molecular_spin_ham_restricted(
            h1e, h2e, nuclear_repulsion)

        # Dump obi and `tbi` to binary file.
        #obi.`astype`(complex).`tofile`(f'{filename}_one_body.`dat`')
        #`tbi`.`astype`(complex).`tofile`(f'{filename}_two_body.`dat`')

        # Compute the PE contribution to the Hamiltonian
        dm = mf_pe.make_rdm1()

        mype = PolEmbed(mol, potfile)
        E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)

        # convert V_pe from atomic orbital to molecular orbital representation
        V_pe_mo = reduce(np.dot, (mf_pe.mo_coeff.T, V_pe, mf_pe.mo_coeff))

        obi_pe = generate_pe_spin_ham_restricted(V_pe_mo)
        #obi_`pe`.`astype`(complex).`tofile`(f'{filename}_`pe`_one_body.`dat`')

        #`metadata = {'num_electrons':nelec, 'num_orbitals':norb, 'nuclear_energy':e_nn, 'PE_energy':E_pe, 'HF_energy':mf_pe.e_tot}`
        #`with open(f'{filename}_metadata.json', 'w') as f:`
        #`    json.dump(metadata, f)`

        return (obi, tbi, nuclear_repulsion, obi_pe, nelec, norb, ferm_ham,
                energies)

    else:
        if integrals_natorb:
            mc = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
            h1e_cas, ecore = mc.get_h1eff(natorbs)
            h2e_cas = mc.get_h2eff(natorbs)
            h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
            h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')

            obi, tbi, core_energy, ferm_ham = generate_molecular_spin_ham_restricted(
                h1e_cas, h2e_cas, ecore)

            # `Dump obi and tbi to binary file.`
            #`obi.astype(complex).tofile(f'{filename}_one_body.dat')`
            #`tbi.astype(complex).tofile(f'{filename}_two_body.dat')`

            if casci:

                dm = mcscf.make_rdm1(mycasci)
                mype = PolEmbed(mol, potfile)
                E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
                #convert from `ao` to mo

                #`V_pe_mo=reduce(np.dot, (mf_pe.mo_coeff.T, V_pe, mf_pe.mo_coeff))`
                V_pe_mo = reduce(np.dot, (natorbs.T, V_pe, natorbs))

                V_pe_cas = V_pe_mo[mycasci.ncore:mycasci.ncore + mycasci.ncas,
                                   mycasci.ncore:mycasci.ncore + mycasci.ncas]

                obi_pe = generate_pe_spin_ham_restricted(V_pe_cas)
                #`obi_pe.astype(complex).tofile(f'{filename}_pe_one_body.dat')`

                #`metadata = {'num_electrons':nele_cas, 'num_orbitals':norb_cas, 'core_energy':ecore, 'PE_energy':E_pe,'HF_energy':mf_pe.e_tot}`
                #`with open(f'{filename}_metadata.json', 'w') as f:`
                #`    json.dump(metadata, f)`

                return (obi, tbi, ecore, obi_pe, nele_cas, norb_cas, ferm_ham,
                        energies)

            else:
                raise ValueError('You should use casci=True.')

        elif integrals_casscf:
            if casscf:
                h1e_cas, ecore = mycas.get_h1eff()
                h2e_cas = mycas.get_h2eff()
                h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')
            else:
                raise ValueError(
                    "WARN: You need to run casscf. Use casscf=True.")
            obi, tbi, core_energy, ferm_ham = generate_molecular_spin_ham_restricted(
                h1e_cas, h2e_cas, ecore)

            # `Dump obi and tbi to binary file.`
            #`obi.astype(complex).tofile(f'{filename}_one_body.dat')`
            #`tbi.astype(complex).tofile(f'{filename}_two_body.dat')`

            dm = mcscf.make_rdm1(mycas)
            # Compute the PE contribution to the Hamiltonian
            mype = PolEmbed(mol, potfile)
            E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
            #convert from `ao` to `mo`
            V_pe_mo = reduce(np.dot, (mycas.mo_coeff.T, V_pe, mycas.mo_coeff))

            V_pe_cas = V_pe_mo[mycas.ncore:mycas.ncore + mycas.ncas,
                               mycas.ncore:mycas.ncore + mycas.ncas]
            obi_pe = generate_pe_spin_ham_restricted(V_pe_cas)
            #`obi_pe.astype(complex).tofile(f'{filename}_pe_one_body.dat')`

            #`metadata = {'num_electrons':nele_cas, 'num_orbitals':norb_cas, 'core_energy':ecore, 'PE_energy':E_pe,'HF_energy':mf_pe.e_tot}`
            #`with open(f'{filename}_metadata.json', 'w') as f:`
            #`    json.dump(metadata, f)`

            return (obi, tbi, ecore, obi_pe, nele_cas, norb_cas, ferm_ham,
                    energies)

        else:
            mc = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
            h1e_cas, ecore = mc.get_h1eff(mf_pe.mo_coeff)
            h2e_cas = mc.get_h2eff(mf_pe.mo_coeff)
            h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
            h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')
            obi, tbi, core_energy, ferm_ham = generate_molecular_spin_ham_restricted(
                h1e_cas, h2e_cas, ecore)

            #` Dump obi and tbi to binary file.`
            #`obi.astype(complex).tofile(f'{filename}_one_body.dat')`
            #`tbi.astype(complex).tofile(f'{filename}_two_body.dat')`

            dm = mf_pe.make_rdm1()
            # Compute the PE contribution to the Hamiltonian
            mype = PolEmbed(mol, potfile)
            E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
            #convert from `ao` to `mo`
            V_pe_mo = reduce(np.dot, (mf_pe.mo_coeff.T, V_pe, mf_pe.mo_coeff))

            V_pe_cas = V_pe_mo[mc.ncore:mc.ncore + mc.ncas,
                               mc.ncore:mc.ncore + mc.ncas]
            obi_pe = generate_pe_spin_ham_restricted(V_pe_cas)
            #`obi_pe.astype(complex).tofile(f'{filename}_pe_one_body.dat')`

            #`metadata = {'num_electrons':nele_cas, 'num_orbitals':norb_cas, 'core_energy':ecore, 'PE_energy':E_pe,'HF_energy':mf_pe.e_tot}`
            #`with open(f'{filename}_metadata.json', 'w') as f:`
            #`    json.dump(metadata, f)`

            return (obi, tbi, ecore, obi_pe, nele_cas, norb_cas, ferm_ham,
                    energies)
