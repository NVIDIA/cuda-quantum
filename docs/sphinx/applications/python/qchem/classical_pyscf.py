# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import json
from functools import reduce

try:
    import numpy as np
except ValueError:
    print('numpy should be installed.')

try:
    from pyscf import gto, scf, cc, ao2mo, mp, mcscf, fci
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
            temp = str(h1e[p, q]) + ' a_' + str(p) + '^dagger ' + 'a_' + str(q)
            ferm_ham.append(temp)
            one_body_coeff[2 * p + 1, 2 * q + 1] = h1e[p, q]
            temp = str(h1e[p, q]) + ' b_' + str(p) + '^dagger ' + 'b_' + str(q)
            ferm_ham.append(temp)

            for r in range(nqubits // 2):
                for s in range(nqubits // 2):

                    # Same spin (`aaaa`, `bbbbb`) <a|a><a|a>, <b|b><b|b>
                    two_body_coeff[2 * p, 2 * q, 2 * r,
                                   2 * s] = 0.5 * h2e[p, q, r, s]
                    temp = str(0.5 * h2e[p, q, r, s]) + ' a_' + str(
                        p) + '^dagger ' + 'a_' + str(
                            q) + '^dagger ' + 'a_' + str(r) + ' a_' + str(s)
                    ferm_ham.append(temp)
                    two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                   2 * s + 1] = 0.5 * h2e[p, q, r, s]
                    temp = str(0.5 * h2e[p, q, r, s]) + ' b_' + str(
                        p) + '^dagger ' + 'b_' + str(
                            q) + '^dagger ' + 'b_' + str(r) + ' b_' + str(s)
                    ferm_ham.append(temp)

                    # Mixed spin(`abab`, `baba`) <a|a><b|b>, <b|b><a|a>
                    #<a|b>= 0 (orthogonal)
                    two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                   2 * s] = 0.5 * h2e[p, q, r, s]
                    temp = str(0.5 * h2e[p, q, r, s]) + ' a_' + str(
                        p) + '^dagger ' + 'a_' + str(
                            q) + '^dagger ' + 'b_' + str(r) + ' b_' + str(s)
                    ferm_ham.append(temp)
                    two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                   2 * s + 1] = 0.5 * h2e[p, q, r, s]
                    temp = str(0.5 * h2e[p, q, r, s]) + ' b_' + str(
                        p) + '^dagger ' + 'b_' + str(
                            q) + '^dagger ' + 'a_' + str(r) + ' a_' + str(s)
                    ferm_ham.append(temp)

    full_hamiltonian = " + ".join(ferm_ham)

    return one_body_coeff, two_body_coeff, ecore, full_hamiltonian


####################################################################

# A- Gas phase simulation
#############################
## Beginning of simulation
#############################

def get_mol_hamiltonian(xyz:str, spin:int, charge: int, basis:str, symmetry:bool = False, memory:float = 4000, cycles:int = 100, \
                              initguess:str = 'minao', nele_cas = None, norb_cas = None, MP2:bool = False, natorb:bool = False,\
                              casci:bool = False, ccsd:bool = False, casscf:bool = False, integrals_natorb:bool = False, \
                                integrals_casscf:bool = False, viz_orb:bool = False, verbose:bool = False):
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

        norb = myhf.mo_coeff[0].shape[1]
        if verbose:
            print('[pyscf] Total number of orbitals = ', norb)

    nelec = mol.nelectron
    if verbose:
        print('[pyscf] Total number of electrons = ', nelec)
    if verbose:
        print('[pyscf] HF energy = ', myhf.e_tot)
    if viz_orb:
        molden.from_mo(mol, filename + '_HF_molorb.molden', myhf.mo_coeff)

    ##########################
    # MP2
    ##########################
    if MP2:

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

    #######################################
    # CASCI if active space is defined
    # FCI if the active space is None
    ######################################
    if casci:

        if nele_cas is None:
            myfci = fci.FCI(myhf)
            result = myfci.kernel()
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
                mycasci_mo = mcscf.CASCI(myhf, norb_cas, nele_cas)
                mycasci_mo.kernel()
                if verbose:
                    print('[pyscf] R-CASCI energy using molecular orbitals= ',
                          mycasci_mo.e_tot)

    ########################
    # CCSD
    ########################
    if ccsd:

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

    #########################
    # CASSCF
    #########################
    if casscf:
        if nele_cas is None:
            raise ValueError("WARN: You should define the active space.")

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

    ###################################
    # CASCI: `FCI` of the active space
    ##################################
    if casci and casscf:

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
                h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')

        elif integrals_casscf:
            if casscf:
                h1e_cas, ecore = mycas.get_h1eff()
                h2e_cas = mycas.get_h2eff()
                h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')
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
        return (obi, tbi, e_nn, nelec, norb, ferm_ham)

    else:
        # `metadata = {'num_electrons_cas': nele_cas, 'num_orbitals_cas': norb_cas, 'core_energy': ecore, 'hf_energy': myhf.e_tot}`
        # with open(f'{filename}_metadata.`json`', 'w') as f:
        #        `json`.dump(metadata, f)
        return (obi, tbi, ecore, nele_cas, norb_cas, ferm_ham)


#######################################################
