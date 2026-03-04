import numpy as np
import random
from scipy import stats
from scipy.stats import skewnorm
from collections import Counter, defaultdict

class ReactionMixin:
    """
    Handles reaction selection and execution for kinetic Monte Carlo simulations.
    Uses 16-parameter rate constant system:
    - C1, C2, C3, C4: Individual ads/des rates
    - C5+: Internal/terminal ads/des rates
    - dMC (dehydrogenation): terminal (C2/C3), internal+terminal (C4+)
    - Cracking: terminal (C2/C3), internal+terminal (C4+)
    """
    
    def select_reaction(self, site_counts):
        # Calculate rates for all possible reactions
        rates = {}
        
        # Adsorption rates
        for key in ['ads_c2', 'ads_c3', 'ads_c4', 'ads_c5plus_internal', 'ads_c5plus_terminal']:
            count = site_counts.get(key, 0)
            if count > 0:
                # Adsorption affected by surface coverage
                k = self.k_const[key]
                rates[key] = k * count * (1 - self.theta)
        
        # Desorption rates
        for key in ['des_c1', 'des_c2', 'des_c3', 'des_c4', 'des_c5plus_internal', 'des_c5plus_terminal']:
            count = site_counts.get(key, 0)
            if count > 0:
                k = self.k_const[key]
                rates[key] = k * count
        
        # Double M-C formation (dehydrogenation) rates
        for key in ['dmc_c2_terminal', 'dmc_c3_terminal', 'dmc_c4_internal', 'dmc_c4_terminal',
                    'dmc_c5plus_internal', 'dmc_c5plus_terminal']:
            count = site_counts.get(key, 0)
            if count > 0:
                # Map to appropriate rate constant
                if 'internal' in key:
                    k = self.k_const['dmc_internal']
                else:
                    k = self.k_const['dmc_terminal']
                rates[key] = k * count
        
        # Cracking rates
        for key in ['crk_c2_terminal', 'crk_c3_terminal', 'crk_c4_internal', 'crk_c4_terminal',
                    'crk_c5plus_internal', 'crk_c5plus_terminal']:
            count = site_counts.get(key, 0)
            if count > 0:
                # Map to appropriate rate constant
                if 'internal' in key:
                    k = self.k_const['crk_internal']
                else:
                    k = self.k_const['crk_terminal']
                rates[key] = k * count
        
        # Total rate
        R = sum(rates.values())
        
        # No reactions possible
        if R == 0:
            return None, 0
        
        # Select reaction using KMC algorithm
        u1 = np.random.rand()
        u2 = np.random.rand()
        
        # Create cumulative distribution
        reaction_keys = list(rates.keys())
        rate_values = np.array([rates[k] for k in reaction_keys])
        cum = np.cumsum(rate_values) / R #normalization of getting CDF [p1,p1+p2,...,1]
        
        # idx = np.argmax(cum >= u1) 
        # cum < u1 gives a True up to the last bin like cum=[0.2, 0.5, 0.7, 1.0], u1 = 0.65 -> cum < u1 [T,T,F,F] where always returns the 0(first T=1)
        # Select reaction: find the FIRST index where cum >= u1
        idx = np.searchsorted(cum, u1, side='right') 
        selected_reaction = reaction_keys[idx]
        
        # Time increment
        dt = -np.log(u2) / R 
        
        return selected_reaction, dt #reaction_key, dt
    
    def perform_reaction(self, reaction_key):
        """
        Execute the selected reaction.
        
        Args:
            reaction_key (str): Reaction identifier (e.g., 'ads_c1', 'crk_c4_internal')
        
        Returns:
            bool: True if reaction succeeded, False otherwise
        """
        # Parse reaction type
        if reaction_key.startswith('ads_'):
            return self.perform_adsorption(reaction_key)
        elif reaction_key.startswith('des_'):
            return self.perform_desorption(reaction_key)
        elif reaction_key.startswith('dmc_'):
            return self.perform_dmc_formation(reaction_key)
        elif reaction_key.startswith('crk_'):
            return self.perform_cracking(reaction_key)
        else:
            return False, None
        
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
    
    def perform_adsorption(self, ads_key, use_normal = True):
        """
        Perform adsorption on vacant sites.
        """
        # Extract chain length and position
        if 'c1' in ads_key:
            target_length = 1
            is_internal = False
        elif 'c2' in ads_key:
            target_length = 2
            is_internal = False
        elif 'c3' in ads_key:
            target_length = 3
            is_internal = False
        elif 'c4' in ads_key:
            target_length = 4
            is_internal = False
        else:  # c5plus
            target_length = 5
            is_internal = 'internal' in ads_key
        
        # Collect adsorption sites
        adsorption_sites = []
        
        for start, end in self.chains:
            chain_length = end - start
            chain_segment = self.carbon_array[start:end]
            
            # Check chain length match
            if target_length <= 4:
                if chain_length != target_length:
                    continue #skips this iteration, go to next chain
            else:  # C5+
                if chain_length < 5:
                    continue
            
            # Only adsorb on completely vacant chains
            if np.any(chain_segment == 1):
                continue
            
            # Collect appropriate sites
            if target_length <= 4:
                # C1-C4: all vacant sites
                sites = [start + i for i in range(chain_length) if chain_segment[i] == 0]
                adsorption_sites.extend(sites)
            else:  # C5+
                if is_internal:
                    internal_candidates = [start + i for i in range(1, chain_length - 1)]
                    # Use extend because sample_adsorption_site now always returns a list
                    selected_sites = self.sample_adsorption_site(
                        internal_candidates, start, chain_length, use_normal=use_normal
                    )
                    adsorption_sites.extend(selected_sites)
                else:
                    # Terminal positions (0 and n-1)
                    if chain_segment[0] == 0:
                        adsorption_sites.append(start)
                    if chain_segment[-1] == 0:
                        adsorption_sites.append(end - 1)
        
        if not adsorption_sites:
            return False, None
        
        # Perform adsorption
        
        site = random.choice(adsorption_sites)
        self.carbon_array[site] = 1
        
        return True, None
    
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