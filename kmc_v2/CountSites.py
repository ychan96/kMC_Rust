import numpy as np
from collections import defaultdict


class ConfigMixin:
    """
    Mixin for chain identification and reaction-site counting.

    Counting output (from update_configuration) is keyed by current fragment
    length N so that reactions.py can compute one Arrhenius rate per (type, N)
    and multiply by the site count — no hardcoded C1/C2/.../C5+ buckets.

    Rate equations (from slide):
        k_ads(N)        = A_ads  * exp(-(E0_ads - alpha_vdw * N) / kT)
        k_d(N)          = A_d    * exp(-(E0_d   + alpha_vdw * N) / kT)
        k_dMC(N, pos)   = A_base * exp(-(E_dMC  + beta_int * is_internal) / kT)
        k_crk(N, pos)   = A_base * exp(-(E_crk  + beta_int * is_internal) / kT)

    Output structure of update_configuration():
        {
            'adsorption': { N: int },
            'desorption': { N: int },
            'dmc':        { N: {'terminal': int, 'internal': int} },
            'cracking':   { N: {'terminal': int, 'internal': int} },
        }

    Relies on BaseKineticMC arrays:
        carbon_array[j]   : 0=free, 1=adsorbed
        chain_array[i]    : 0=break/boundary, 1=bonded  (length N_total + 1)
        occupancy[i]      : 0=vacant, 1=single M-C, 2=dMC
        carbon_to_site[j] : surface site index for carbon j (-1 if free)
        self.surface      : CatalystSurface (provides get_c_neighbors)
    """

    # ------------------------------------------------------------------
    # Chain identification
    # ------------------------------------------------------------------

    def _identify_chains(self):
        """
        Scan chain_array for breaks (value 0) to produce contiguous fragments.

        chain_array has length N_total + 1.
        chain_array[i] == 1 means carbon[i-1] and carbon[i] are bonded.
        Boundaries (index 0 and N_total) are always 0.

        Returns:
            List of (start, end) tuples — carbon_array[start:end] is one fragment.
        """
        chains      = []
        chain_start = 0
        n_carbons   = len(self.carbon_array)

        for i in range(1, n_carbons + 1):
            if self.chain_array[i] == 0:
                if i > chain_start:
                    chains.append((chain_start, i))
                chain_start = i

        return chains

    # ------------------------------------------------------------------
    # Top-level counter
    # ------------------------------------------------------------------

    def update_configuration(self):
        """
        Count available reaction sites for every fragment, grouped by
        reaction type and current fragment length N.

        Returns
        -------
        counts : dict
            counts['adsorption'][N]           -> int   (free carbons in length-N fragments)
            counts['desorption'][N]           -> int   (single-MC carbons in length-N fragments)
            counts['dmc'][N]['terminal']      -> int
            counts['dmc'][N]['internal']      -> int
            counts['cracking'][N]['terminal'] -> int
            counts['cracking'][N]['internal'] -> int
        """
        counts = {
            'adsorption': defaultdict(int),
            'desorption': defaultdict(int),
            'dmc':        defaultdict(lambda: {'terminal': 0, 'internal': 0}),
            'cracking':   defaultdict(lambda: {'terminal': 0, 'internal': 0}),
        }

        n_vacant_c_sites = int(np.sum(self.occupancy == 0))

        for start, end in self.chains:
            N   = end - start
            seg = self.carbon_array[start:end]   # view — no copy

            self._count_adsorption(seg, N, counts, n_vacant_c_sites)
            self._count_desorption(seg, N, counts)
            self._count_dmc(seg, N, counts, n_vacant_c_sites)
            self._count_cracking(seg, N, counts)

        return counts

    # ------------------------------------------------------------------
    # Adsorption  —  k_ads(N) = A_ads * exp(-(E0_ads - alpha_vdw * N) / kT)
    # ------------------------------------------------------------------

    def _count_adsorption(self, seg, N, counts, n_vacant_c_sites):
        """
        Count free carbons (carbon_array == 0) in this fragment.

        Every free carbon is a candidate adsorption site; the rate equation
        encodes the N dependence via alpha_vdw.  We gate on surface vacancy
        globally — if no vacant C sites exist, nothing can adsorb.

        Partial adsorption is allowed: a fragment with some adsorbed carbons
        can still have other free carbons that adsorb independently.
        """
        if n_vacant_c_sites == 0: #no active sites are available on the surface
            return
        if self.n_vacant_h_sites == 0: #no vacant H sites are available on the surface
            return
        if np.any(seg == 1): # no re-adsorption on a fragment that already has adsorbed carbons (competitive Langmuir logic)
            return

        counts['adsorption'][N] += int(np.sum(seg == 0))

    # ------------------------------------------------------------------
    # Desorption  —  k_d(N) = A_d * exp(-(E0_d + alpha_vdw * N) / kT) 
    # ------------------------------------------------------------------

    def _count_desorption(self, seg, N, counts):
        """
        Count carbons in single M-C state (carbon_array == 1, occupancy == 1).

        dMC carbons (occupancy == 2) are NOT eligible for direct desorption —
        they must crack first.  We verify via occupancy to distinguish the two.
        """
        if int(np.sum(seg == 1)) == 1:
            counts['desorption'][N] += 1

    # ------------------------------------------------------------------
    # dMC formation  —  k_dMC(N, pos) = A_base * exp(-(E_dMC + beta_int * is_internal) / kT)
    # ------------------------------------------------------------------

    def _count_dmc(self, seg, N, counts, n_vacant_c_sites):
        """
        Count dMC formation opportunities in this fragment.

        Requirements:
            1. Exactly one carbon in the fragment is in single M-C (occupancy == 1).
            2. That carbon has at least one free chain-neighbor (carbon_array == 0).
            3. A vacant neighboring C site exists on the surface.
        Position label:
            'terminal' if the single M-C carbon is at position 0 or N-1 in the fragment. 
            'internal' otherwise.

        N == 1 is excluded (no chain neighbor).
        """
        #Gate on vacancies and its state — if no vacant H sites or no vacant C sites, dMC formation is impossible
        if self.n_vacant_h_sites == 0:
            return
        if  n_vacant_c_sites == 0:  
            return
        if N == 1:
            return
        if int(np.sum(seg == 1)) != 1: # exactly one carbon must be in single M-C state
            return
        
        idx = np.where(seg == 1)[0][0]

        is_internal =(idx != 0) and (idx != N - 1)

        # check left neighbor
        if idx > 0 and seg[idx - 1] == 0: #this handles the terminal(idx = N-1)
            counts['dmc'][N]['internal' if is_internal else 'terminal'] += 1

        # check right neighbor
        if idx < N - 1 and seg[idx + 1] == 0:
            counts['dmc'][N]['internal' if is_internal else 'terminal'] += 1
        #position label is for reactions.py to apply the correct beta_int penalty

    # ------------------------------------------------------------------
    # Cracking  —  k_crk(N, pos) = A_base * exp(-(E_crk + beta_int * is_internal) / kT)
    # ------------------------------------------------------------------

    def _count_cracking(self, seg, N, counts):
        """
        Count C-C scission opportunities in this fragment.

        Requirements:
            Two adjacent carbons are both adsorbed (seg[i] == 1, seg[i+1] == 1)
            AND at least one of the pair is in dMC state (occupancy == 2),
            confirming the double-bond geometry needed for scission.

        Position label:
            'terminal' if the bond sits at position 0-1 or (N-2)-(N-1).
            'internal' otherwise.

        N == 1 is excluded (no C-C bond).
        """
        # gate on C1 and total attached sites
        if N == 1:
            return
        if int(np.sum(seg == 1)) != 2:
            return
        
        #check for 11 patterns and its position
        for i in range(N - 1):
            if seg[i] == 1 and seg[i + 1] == 1:
                at_terminal = (i == 0) or (i == N - 2)
                key = 'terminal' if at_terminal else 'internal'
                counts['cracking'][N][key] += 1