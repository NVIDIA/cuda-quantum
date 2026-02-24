try:
    import numpy as np
except ValueError:
    print('Numpy should be installed')

#`try:`
#`    from pyscf import gto,scf,solvent`
#`except ValueError:`
#`    print('PySCF should be installed. Use pip install pyscf')`

try:
    import cppe
except ValueError:
    print(
        'cppe library should be installed to use the polaizable embedding model. Use pip install cppe'
    )

#` The CPPE library can be installed via: pip install cppe`
#` You might need to use (apt-get update & apt-get install build-essential -y`
#`                         apt-get install python3-dev) before installing cppe.`
####################################


class PolEmbed:

    def __init__(self, mol, peoptions):

        self.mol = mol

        if isinstance(peoptions, str):
            options = {"potfile": peoptions}
        else:
            options = peoptions

        if not isinstance(options, dict):
            raise TypeError("Options should be a dictionary.")

        self.options = options
        self.cppe_state = self._cppe_state(mol)
        self.potentials = self.cppe_state.potentials
        self.polarizable_coords = np.array([
            site.position
            for site in self.cppe_state.potentials
            if site.is_polarizable
        ])

        self.V_es = None

    def _cppe_state(self, mol):

        cppe_mol = cppe.Molecule()
        for z, coord in zip(mol.atom_charges(), mol.atom_coords()):
            cppe_mol.append(cppe.Atom(z, *coord))

        cppe_state = cppe.CppeState(self.options, cppe_mol)
        cppe_state.calculate_static_energies_and_fields()

        return cppe_state

    def _compute_field_integrals(self, site, moment):
        self.mol.set_rinv_orig(site)
        integral = self.mol.intor("int1e_iprinv") + self.mol.intor(
            "int1e_iprinv").transpose(0, 2, 1)
        op = np.einsum('aij,a->ij', integral, -1.0 * moment)

        return op

    def _compute_field(self, site, dm):

        self.mol.set_rinv_orig(site)
        integral = self.mol.intor("int1e_iprinv") + self.mol.intor(
            "int1e_iprinv").transpose(0, 2, 1)

        return np.einsum('ij,aij->a', dm, integral)

    def _compute_multipole_potential_integrals(self, site, order, moments):

        if order > 2:
            raise NotImplementedError(
                """Multipole potential integrals not implemented for order > 2."""
            )

        self.mol.set_rinv_orig(site)

        # Order 0:
        #if order==0:
        integral0 = self.mol.intor("int1e_rinv")
        es_operator = integral0 * moments[0] * cppe.prefactors(0)

        # Order 1:
        if order > 0:
            integral1 = self.mol.intor("int1e_iprinv") + self.mol.intor(
                "int1e_iprinv").transpose(0, 2, 1)
            es_operator += np.einsum('aij,a->ij', integral1,
                                     moments[1] * cppe.prefactors(1))

        # Order 2:
        if order > 1:
            integral2 = self.mol.intor("int1e_ipiprinv") + self.mol.intor("int1e_ipiprinv").transpose(0, 2, 1) \
                + 2.0 * self.mol.intor("int1e_iprinvip")
            # add the lower triangle to the upper triangle
            integral2[1] += integral2[3]
            integral2[2] += integral2[6]
            integral2[5] += integral2[7]
            integral2[1] *= 0.5
            integral2[2] *= 0.5
            integral2[5] *= 0.5

            es_operator += np.einsum('aij,a->ij',
                                     integral2[[0, 1, 2, 4, 5, 8], :, :],
                                     moments[2] * cppe.prefactors(2))

        return es_operator

    def get_pe_contribution(self, dm, elec_only=False):

        # Step I: Build electrostatic operator
        if self.V_es is None:
            self.V_es = np.zeros((self.mol.nao, self.mol.nao), dtype=np.float64)
            for p in self.potentials:
                moments = []
                for m in p.multipoles:
                    m.remove_trace()
                    moments.append(m.values)
                self.V_es += self._compute_multipole_potential_integrals(
                    p.position, m.k, moments)

        e_static = np.einsum('ij,ij->', self.V_es, dm)
        self.cppe_state.energies["Electrostatic"]["Electronic"] = e_static

        #` Step II: obtain expectation values of elec. field at polarizable sites`
        n_sitecoords = 3 * self.cppe_state.get_polarizable_site_number()
        V_ind = np.zeros((self.mol.nao, self.mol.nao), dtype=np.float64)
        if n_sitecoords:
            current_polsite = 0
            elec_fields = np.zeros(n_sitecoords, dtype=np.float64)
            for p in self.potentials:
                if not p.is_polarizable:
                    continue
                elec_fields_s = self._compute_field(p.position, dm)
                elec_fields[3 * current_polsite:3 * current_polsite +
                            3] = elec_fields_s
                current_polsite += 1

        # Step III: solve induced moments
            self.cppe_state.update_induced_moments(elec_fields, elec_only)
            induced_moments = np.array(self.cppe_state.get_induced_moments())

            # Step IV: build induction operator
            current_polsite = 0
            for p in self.potentials:
                if not p.is_polarizable:
                    continue
                site = p.position
                V_ind += self._compute_field_integrals(
                    site=site,
                    moment=induced_moments[3 *
                                           current_polsite:3 * current_polsite +
                                           3])
                current_polsite += 1

        E_pe = self.cppe_state.total_energy

        if not elec_only:
            V_pe = self.V_es + V_ind
        else:
            V_pe = V_ind
            E_pe = self.cppe_state.energies["Polarization"]["Electronic"]

        return E_pe, V_pe, self.V_es, V_ind
