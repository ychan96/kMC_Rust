import numpy as np
import random
from scipy import stats
from scipy.stats import skewnorm
from collections import Counter, defaultdict

class ReactionMixin:
    """
    Reaction selection using Gillespie algorithm.
    Consumes update_configuration() output:
        counts['adsorption'][N]           -> int
        counts['desorption'][N]           -> int
        counts['dmc'][N]['terminal']      -> int
        counts['dmc'][N]['internal']      -> int
        counts['cracking'][N]['terminal'] -> int
        counts['cracking'][N]['internal'] -> int
    """

    def select_reaction(self, counts):
        rates = {}

        # Adsorption — k_ads(N) = A_ads * exp(-(E0_ads - alpha_vdw * N) / kT)
        for N, n_sites in counts['adsorption'].items():
            if n_sites > 0:
                rates[('adsorption', N, None)] = self.get_rate(N, 'adsorption') * n_sites

        # Desorption — k_d(N) = A_d * exp(-(E0_d + alpha_vdw * N) / kT)
        for N, n_sites in counts['desorption'].items():
            if n_sites > 0:
                rates[('desorption', N, None)] = self.get_rate(N, 'desorption') * n_sites

        # dMC formation — k_dMC(N, pos) = A_base * exp(-(E_dMC + beta_int * is_internal) / kT)
        for N, pos_counts in counts['dmc'].items():
            for pos, n_sites in pos_counts.items():
                if n_sites > 0:
                    is_internal = (pos == 'internal')
                    rates[('dmc', N, pos)] = self.get_rate(N, 'dMC', is_internal) * n_sites

        # Cracking — k_crk(N, pos) = A_base * exp(-(E_crk + beta_int * is_internal) / kT)
        for N, pos_counts in counts['cracking'].items():
            for pos, n_sites in pos_counts.items():
                if n_sites > 0:
                    is_internal = (pos == 'internal')
                    rates[('cracking', N, pos)] = self.get_rate(N, 'cracking', is_internal) * n_sites

        R = sum(rates.values())
        if R == 0:
            return None, 0

        # BKL selection
        reaction_keys = list(rates.keys())
        rate_values   = np.array([rates[k] for k in reaction_keys])
        cum           = np.cumsum(rate_values) / R

        u1 = np.random.rand()
        u2 = np.random.rand()

        idx               = np.searchsorted(cum, u1, side='right')
        selected_reaction = reaction_keys[idx]       # (reaction_type, N, pos)

        dt = -np.log(u2) / R

        return selected_reaction, dt
    
    def perform_reaction(self, reaction_key):
        reaction_type, N, pos = reaction_key

        if reaction_type == 'adsorption':
            return self.perform_adsorption(N)
        elif reaction_type == 'desorption':
            return self.perform_desorption(N)
        elif reaction_type == 'dmc':
            return self.perform_dmc_formation(N, pos)
        elif reaction_type == 'cracking':
            return self.perform_cracking(N, pos)
        else:
            return False
        
    def sample_adsorption_site(self, internal_sites, chain_start, chain_length, use_normal=False):
        """
        Sample one adsorption site from internal_sites.
        Defaults to uniform distribution. If use_normal=True, uses a Gaussian 
        distribution centered at the midpoint.
        """
        if not internal_sites:
            return None

        if use_normal:
            # Gaussian logic
            sigma = chain_length / 12
            mid = chain_start + (chain_length - 1) / 2
            weights = stats.norm.pdf(internal_sites, loc=mid, scale=sigma)
            weights /= weights.sum()
            selected = np.random.choice(internal_sites, p=weights)
            return [selected]
        else:
            # Uniform logic: every site has the same probability
            return internal_sites
    
    def perform_adsorption(self, N):
        # 1. Find all C-N fragments (no adsorbed carbons)
        candidate_fragments = []
        for start, end in self.chains:
            if end - start == N:
                seg = self.carbon_array[start:end]
                if not np.any(seg == 1):
                    candidate_fragments.append(start)

        if not candidate_fragments:
            return False

        # 2. Randomly pick one C-N fragment
        start = np.random.choice(candidate_fragments)
        seg   = self.carbon_array[start:start + N]

        # 3. Pick one carbon via uniform random (all free carbons equally likely)
        free_positions = np.where(seg == 0)[0]
        local_c      = np.random.choice(free_positions)
        global_c  = start + local_c
        is_terminal    = (local_c == 0) or (local_c == N - 1)

        # 4. Find a vacant C site on the surface
        vacant_c_sites = np.where(self.occupancy == 0)[0]
        if len(vacant_c_sites) == 0:
            return False
        site_idx = np.random.choice(vacant_c_sites)

        # 5. Update carbon arrays
        self.carbon_array[global_c]    = 1
        self.hydrogen_array[global_c]  = 0   # 3->0 terminal, 2->0 internal

        # 6. Update C site arrays
        self.occupancy[site_idx]            = 1   # single M-C
        self.carbon_at_site[site_idx]       = global_c
        self.chain_at_site[site_idx]        = N
        self.carbon_to_site[global_c]  = site_idx

        # 7. Scatter released H atoms onto vacant hollow sites
        n_h_released   = 3 if is_terminal else 2
        vacant_h_sites = np.where(self.h_occupancy == 0)[0]
        if len(vacant_h_sites) < n_h_released:
            return False   # not enough H sites — should be gated in count_sites
        chosen_h_sites = np.random.choice(vacant_h_sites, size=n_h_released, replace=False)
        self.h_occupancy[chosen_h_sites] = 1
        
        return True
    def _get_chain_info_for_carbon(self, carbon_idx):
        """
        Get chain length that this carbon belongs to
        self.chains comes from the @property in init.py -> it calls _identify_chains() from count_sites.py
        """
        for start, end in self.chains: 
            if start <= carbon_idx < end:
                return end - start
        return None


    def perform_desorption(self, des_key):
        """
        Perform desorption on a single attached carbon.
        """
        # Extract target parameters
        if 'c1' in des_key:
            target_length = 1
            is_internal = False
        elif 'c2' in des_key:
            target_length = 2
            is_internal = False
        elif 'c3' in des_key:
            target_length = 3
            is_internal = False
        elif 'c4' in des_key:
            target_length = 4
            is_internal = False
        else:  # c5plus
            target_length = 5
            is_internal = 'internal' in des_key
        
        # Collect desorption sites
        desorption_sites = []
        
        for start, end in self.chains:
            chain_length = end - start
            chain_segment = self.carbon_array[start:end]
            
            # Check chain length match
            if target_length <= 4:
                if chain_length != target_length:
                    continue
            else:  # C5+
                if chain_length < 5:
                    continue
            
            # Check exactly one attached
            num_attached = np.sum(chain_segment == 1)
            if num_attached != 1:
                continue
            
            # Check no adjacent 1s
            has_adjacent = np.any((chain_segment[:-1] == 1) & (chain_segment[1:] == 1))
            if has_adjacent:
                continue
            
            # Find attached position
            attached_idx = np.where(chain_segment == 1)[0][0]
            is_terminal_pos = (attached_idx == 0) or (attached_idx == chain_length - 1)
            
            # Check position match
            if target_length <= 4:
                # C1-C4: add site
                desorption_sites.append(start + attached_idx)
            else:  # C5+
                if is_internal and not is_terminal_pos:
                    desorption_sites.append(start + attached_idx)
                elif not is_internal and is_terminal_pos:
                    desorption_sites.append(start + attached_idx)
        
        if not desorption_sites:
            return False, None
        
        # Perform desorption
        site = random.choice(desorption_sites)

        # Get chain info before clearing
        chain_info = self._get_chain_info_for_carbon(site)
        
        self.carbon_array[site] = 0
        
        return True, chain_info  # Return success + info


    
    def perform_dmc_formation(self, dmc_key):
        """
        Perform double M-C bond formation (dehydrogenation).
        
        Args:
            dmc_key (str): e.g., 'dmc_c2_terminal', 'dmc_c5plus_internal'
        """
        # Extract target parameters
        if 'c2' in dmc_key:
            target_length = 2
            is_internal = False
        elif 'c3' in dmc_key:
            target_length = 3
            is_internal = False
        elif 'c4' in dmc_key:
            target_length = 4
            is_internal = 'internal' in dmc_key
        else:  # c5plus
            target_length = 5
            is_internal = 'internal' in dmc_key
        
        # Collect dMC formation sites
        dmc_sites = []
        
        for start, end in self.chains:
            chain_length = end - start
            chain_segment = self.carbon_array[start:end]
            
            # Check chain length match
            if target_length <= 4:
                if chain_length != target_length:
                    continue
            else:  # C5+
                if chain_length < 5:
                    continue
            
            # Check exactly one attached
            num_attached = np.sum(chain_segment == 1)
            if num_attached != 1:
                continue
            
            # Find attached position
            attached_idx = np.where(chain_segment == 1)[0][0]
            
            # Check vacant neighbors
            left_vacant = attached_idx > 0 and chain_segment[attached_idx - 1] == 0
            right_vacant = attached_idx < chain_length - 1 and chain_segment[attached_idx + 1] == 0
            
            if not (left_vacant or right_vacant):
                continue
            
            # For C2, C3: all positions are "terminal"
            if target_length in [2, 3]:
                if left_vacant:
                    dmc_sites.append(start + attached_idx - 1)
                if right_vacant:
                    dmc_sites.append(start + attached_idx + 1)
            else: 
                # C4+, C5+: check internal vs terminal
                if left_vacant:
                    # Bond at (attached_idx-1, attached_idx)
                    bond_at_end = (attached_idx - 1 == 0) or (attached_idx == chain_length - 1)
                    #..001 -> left vacant, but terminal
                    if is_internal != bond_at_end:
                        dmc_sites.append(start + attached_idx - 1)
                
                if right_vacant:
                    # Bond at (attached_idx, attached_idx+1)
                    bond_at_end = (attached_idx == 0) or (attached_idx == chain_length - 2)
                    #100.. / ..010 -> right vacant True, but terminal
                    if is_internal != bond_at_end:
                        dmc_sites.append(start + attached_idx + 1)
        
        if not dmc_sites:
            return False, None
        
        # Perform dMC formation
        site = random.choice(dmc_sites)

        #get chain info before setting
        chain_info = self._get_chain_info_for_carbon(site)

        self.carbon_array[site] = 1
        
        return True, chain_info
    
    def perform_cracking(self, crk_key):
        """
        Perform C-C bond scission (cracking).
        
        Args:
            crk_key (str): e.g., 'crk_c2_terminal', 'crk_c4_internal'
        """
        # Extract target parameters
        if 'c2' in crk_key:
            target_length = 2
            is_internal = False
        elif 'c3' in crk_key:
            target_length = 3
            is_internal = False
        elif 'c4' in crk_key:
            target_length = 4
            is_internal = 'internal' in crk_key
        else:  # c5+
            target_length = 5
            is_internal = 'internal' in crk_key
        
        # Collect cracking sites
        cracking_sites = []
        
        for start, end in self.chains:
            chain_length = end - start
            chain_segment = self.carbon_array[start:end]
            
            # Check chain length match
            if target_length <= 4:
                if chain_length != target_length:
                    continue
            else:  # C5+
                if chain_length < 5:
                    continue
            
            # Find 11 patterns (double M-C bonds)
            for i in range(chain_length - 1):
                if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                    # Bond at position i (between carbons i and i+1)
                    bond_position = i
                    
                    # For C2, C3: all bonds are "terminal"
                    if target_length in [2, 3]:
                        # Break bond by setting chain_array
                        cracking_sites.append(start + i + 1)  # Position in chain_array
                    else:  # C4+
                        # Check if bond is terminal or internal
                        bond_at_end = (bond_position == 0) or (bond_position == chain_length - 2)
                        
                        if is_internal != bond_at_end:
                            cracking_sites.append(start + i + 1)

        if not cracking_sites:
            return False, None
        
        # Perform cracking
        chain_index = random.choice(cracking_sites)

        # Get chain info before breaking
        for start, end in self.chains:
            if start < chain_index <= end:
                original_len = end - start
                # Find cracking position to determine fragments
                chain_segment = self.carbon_array[start:end]

                #find the 11 pattern
                for i in range(len(chain_segment) - 1):
                    if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                        #find lengths of two fragments
                        frag1 = i + 1
                        frag2 = original_len - frag1
                        break #breaks once the 11 pattern is found (no need to keep searching)
                break
        self.chain_array[chain_index] = 0
        
        # Invalidate chain cache
        self.invalidate_chains()
        chain_info = (original_len, frag1, frag2)
        
        return True, chain_info