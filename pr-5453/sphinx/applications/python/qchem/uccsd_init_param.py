try:
    from pyscf import gto, scf, cc, mcscf, mp, solvent
except ValueError:
    print('PySCF should be installed to use pyscf. Use pip install pyscf')

import numpy as np


###################################
def get_thetas_unpack_restricted(singles, doubles, n_occupied, n_virtual):

    theta_1 = np.zeros((2 * n_occupied, 2 * n_virtual))
    theta_2 = np.zeros(
        (2 * n_occupied, 2 * n_occupied, 2 * n_virtual, 2 * n_virtual))

    for p in range(n_occupied):
        for q in range(n_virtual):
            theta_1[2 * p, 2 * q] = singles[p, q]
            theta_1[2 * p + 1, 2 * q + 1] = singles[p, q]

    for p in range(n_occupied):
        for q in range(n_occupied):
            for r in range(n_virtual):
                for s in range(n_virtual):
                    theta_2[2 * p, 2 * q, 2 * s, 2 * r] = doubles[p, q, r, s]
                    theta_2[2 * p + 1, 2 * q + 1, 2 * r + 1,
                            2 * s + 1] = doubles[p, q, r, s]
                    theta_2[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = doubles[p, q,
                                                                          r, s]
                    theta_2[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = doubles[p, q,
                                                                          r, s]

    return theta_1, theta_2


##############################################
def uccsd_get_amplitude(single_theta, double_theta, n_electrons, n_qubits, UR):

    # compute a packed list that contains relevant amplitude for the UCCSD.
    # We store first single excitation amplitudes then double excitation amplitudes.

    # `single_amplitude: [N_occ, N_virt] array storing the single excitation amplitude.`
    # `double_amplitude: [N_occ, N_occ, N_virt, N_virt] array storing double excitation amplitude.`

    if n_qubits % 2 != 0:
        raise ValueError(
            "Total number of spin molecular orbitals (number of qubits) should be even."
        )
    else:
        n_spatial_orbitals = n_qubits // 2

    #if n_electrons%2!=0 and spin>0:
    if UR:
        t1a, t1b = single_theta
        t2aa, t2ab, t2bb = double_theta
        t2ab = np.asarray(t2ab.transpose(0, 1, 3, 2), order='C')

        n_occupied_alpha, n_virtual_alpha = t1a.shape
        n_occupied_beta, n_virtual_beta = t1b.shape

        singles_alpha = []
        singles_beta = []
        doubles_mixed = []
        doubles_alpha = []
        doubles_beta = []

        for p in range(n_occupied_alpha):
            for q in range(n_virtual_alpha):
                singles_alpha.append(t1a[p, q])

        for p in range(n_occupied_beta):
            for q in range(n_virtual_beta):
                singles_beta.append(t1b[p, q])

        for p in range(n_occupied_alpha):
            for q in range(n_occupied_beta):
                for r in range(n_virtual_beta):
                    for s in range(n_virtual_alpha):
                        doubles_mixed.append(t2ab[p, q, r, s])

        for p in range(n_occupied_alpha - 1):
            for q in range(p + 1, n_occupied_alpha):
                for r in range(n_virtual_alpha - 1):
                    for s in range(r + 1, n_virtual_alpha):
                        doubles_alpha.append(t2aa[p, q, r, s])

        for p in range(n_occupied_beta - 1):
            for q in range(p + 1, n_occupied_beta):
                for r in range(n_virtual_beta - 1):
                    for s in range(r + 1, n_virtual_beta):
                        doubles_alpha.append(t2bb[p, q, r, s])

    #`elif n_electrons%2==0 and spin==0:`
    else:
        n_occupied = n_electrons // 2
        n_virtual = n_spatial_orbitals - n_occupied

        single_amplitude, double_amplitude = get_thetas_unpack_restricted(
            single_theta, double_theta, n_occupied, n_virtual)

        singles_alpha = []
        singles_beta = []
        doubles_mixed = []
        doubles_alpha = []
        doubles_beta = []

        occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
        virtual_alpha_indices = [i * 2 for i in range(n_virtual)]

        occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied)]
        virtual_beta_indices = [i * 2 + 1 for i in range(n_virtual)]

        # Same spin single excitation
        for p in occupied_alpha_indices:
            for q in virtual_alpha_indices:
                singles_alpha.append(single_amplitude[p, q])

        for p in occupied_beta_indices:
            for q in virtual_beta_indices:
                singles_beta.append(single_amplitude[p, q])

        #Mixed spin double excitation
        for p in occupied_alpha_indices:
            for q in occupied_beta_indices:
                for r in virtual_beta_indices:
                    for s in virtual_alpha_indices:
                        doubles_mixed.append(double_amplitude[p, q, r, s])

        # same spin double excitation
        n_occ_alpha = len(occupied_alpha_indices)
        n_occ_beta = len(occupied_beta_indices)
        n_virt_alpha = len(virtual_alpha_indices)
        n_virt_beta = len(virtual_beta_indices)

        for p in range(n_occ_alpha - 1):
            for q in range(p + 1, n_occ_alpha):
                for r in range(n_virt_alpha - 1):
                    for s in range(r + 1, n_virt_alpha):

                        # Same spin: all alpha
                        doubles_alpha.append(double_amplitude[occupied_alpha_indices[p], occupied_alpha_indices[q],\
                                        virtual_alpha_indices[r], virtual_alpha_indices[s]])

        for p in range(n_occ_beta - 1):
            for q in range(p + 1, n_occ_beta):
                for r in range(n_virt_beta - 1):
                    for s in range(r + 1, n_virt_beta):

                        # Same spin: all beta
                        doubles_beta.append(double_amplitude[occupied_beta_indices[p], occupied_beta_indices[q],\
                                        virtual_beta_indices[r], virtual_beta_indices[s]])

    return singles_alpha + singles_beta + doubles_mixed + doubles_alpha + doubles_beta


##############################

def get_parameters(xyz:str, spin:int, charge: int, basis:str, symmetry:bool=False, memory:float=4000,cycles:int=100, \
                              initguess:str='minao', UR:bool=False, nele_cas=None, norb_cas=None, MP2:bool=False, natorb:bool=False,\
                              ccsd:bool=False, without_solvent:bool=True, potfile:str=None, verbose:bool=False):

    if verbose:
        print(
            '[pyscf] Computing initial guess parameters using the classical CCSD'
        )

    filename = xyz.split('.')[0]
    mol = gto.M(atom=xyz,
                spin=spin,
                charge=charge,
                basis=basis,
                max_memory=memory,
                symmetry=symmetry,
                output=filename + '-pyscf.log',
                verbose=4)

    if without_solvent:
        if UR:
            myhf = scf.UHF(mol)
            myhf.max_cycle = cycles
            myhf.chkfile = filename + '-pyscf.chk'
            myhf.init_guess = initguess
            myhf.kernel()

            norb = myhf.mo_coeff[0].shape[1]
            if verbose:
                print('[pyscf] Total number of alpha molecular orbitals = ',
                      norb)
            norb = myhf.mo_coeff[0].shape[0]
            if verbose:
                print('[pyscf] Total number of beta molecular orbitals = ',
                      norb)

        else:
            myhf = scf.RHF(mol)
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

        ##########################
        # MP2
        ##########################
        if MP2:

            if UR:
                mymp = mp.UMP2(myhf)
                mp_ecorr, mp_t2 = mymp.kernel()
                if verbose:
                    print('[pyscf] UR-MP2 energy= ', mymp.e_tot)

                if natorb:
                    # Compute natural orbitals
                    dma, dmb = mymp.make_rdm1()
                    noon_a, U_a = np.linalg.eigh(dma)
                    noon_b, U_b = np.linalg.eigh(dmb)
                    noon_a = np.flip(noon_a)
                    noon_b = np.flip(noon_b)

                    if verbose:
                        print(
                            'Natural orbital (alpha orbitals) occupation number from UR-MP2: '
                        )
                    if verbose:
                        print(noon_a)
                    if verbose:
                        print(
                            'Natural orbital (beta orbitals) occupation number from UR-MP2: '
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

                    if natorb:
                        # Compute natural orbitals
                        noons, natorbs = mcscf.addons.make_natural_orbitals(
                            mymp)
                        if verbose:
                            print(
                                'Natural orbital occupation number from R-MP2: '
                            )
                        if verbose:
                            print(noons)

        if ccsd:

            if UR:

                mc = mcscf.UCASCI(myhf, norb_cas, nele_cas)
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

            else:
                if nele_cas is None:
                    mycc = cc.CCSD(myhf)
                    mycc.max_cycle = cycles
                    mycc.kernel()
                    if verbose:
                        print('[pyscf] Total R-CCSD energy = ', mycc.e_tot)
                    qubits_num = 2 * norb
                    init_params = uccsd_get_amplitude(mycc.t1, mycc.t2, nelec,
                                                      qubits_num, UR)
                    #`np.array(init_params).tofile(f'{filename}_params.dat')`
                    return init_params

                else:
                    mc = mcscf.CASCI(myhf, norb_cas, nele_cas)
                    frozen = []
                    frozen += [y for y in range(0, mc.ncore)]
                    frozen += [
                        y for y in range(mc.ncore +
                                         norb_cas, len(myhf.mo_coeff))
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

                    qubits_num = 2 * norb_cas
                    init_params = uccsd_get_amplitude(mycc.t1, mycc.t2,
                                                      nele_cas, qubits_num, UR)
                    #`np.array(init_params).tofile(f'{filename}_params.dat')`
                    return init_params

    # With solvent
    else:
        nelec = mol.nelectron
        if verbose:
            print('[Pyscf] Total number of electrons = ', nelec)

        # HF with PE model.
        mf_pe = scf.RHF(mol)
        mf_pe.init_guess = initguess
        mf_pe = solvent.PE(mf_pe, potfile).run()
        norb = mf_pe.mo_coeff.shape[1]
        if verbose:
            print('[Pyscf] Total number of orbitals = ', norb)
        if verbose:
            print('[Pyscf] Total HF energy with solvent:', mf_pe.e_tot)
        if verbose:
            print('[Pyscf] Polarizable embedding energy from HF: ',
                  mf_pe.with_solvent.e)

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

                if natorb:
                    # Compute natural orbitals
                    noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
                    if verbose:
                        print(
                            '[Pyscf] Natural orbital occupation number from R-MP2: '
                        )
                    if verbose:
                        print(noons)

        if ccsd:
            if nele_cas is None:
                mycc = cc.CCSD(mf_pe)
                mycc = solvent.PE(mycc, potfile)
                mycc.run()
                if verbose:
                    print('[Pyscf] Total CCSD energy with solvent: ',
                          mycc.e_tot)
                if verbose:
                    print('[Pyscf] Polarizable embedding energy from CCSD: ',
                          mycc.with_solvent.e)
                qubits_num = 2 * norb
                init_params = uccsd_get_amplitude(mycc.t1, mycc.t2, nelec,
                                                  qubits_num, UR)
                #`np.array(init_params).tofile(f'{filename}_params.dat')```
                return init_params

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
                        print(
                            '[Pyscf] Polarizable embedding energy from CCSD: ',
                            mycc.with_solvent.e)

                qubits_num = 2 * norb_cas
                init_params = uccsd_get_amplitude(mycc.t1, mycc.t2, nele_cas,
                                                  qubits_num, UR)
                #`np.array(init_params).tofile(f'{filename}_params.dat')`
                return init_params
