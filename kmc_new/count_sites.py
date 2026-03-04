import numpy as np
from kmc_new.init import BaseKineticMC

class ConfigMixin:
    def update_configuration(self):
        """
        Count available sites for each reaction type across all chain lengths.
        Returns a dictionary of counts organized by reaction type and chain length.
        """
        counts = {
            # Adsorption sites (by chain length)
            'ads_c1': 0,
            'ads_c2': 0,
            'ads_c3': 0,
            'ads_c4': 0,
            'ads_c5plus_internal': 0,
            'ads_c5plus_terminal': 0,
            
            # Desorption sites (by chain length)
            'des_c1': 0,
            'des_c2': 0,
            'des_c3': 0,
            'des_c4': 0,
            'des_c5plus_internal': 0,
            'des_c5plus_terminal': 0,
            
            # Double M-C formation (dehydrogenation)
            'dmc_c2_terminal': 0,
            'dmc_c3_terminal': 0,
            'dmc_c4_internal': 0,
            'dmc_c4_terminal': 0,
            'dmc_c5plus_internal': 0,
            'dmc_c5plus_terminal': 0,
            
            # Cracking sites
            'crk_c2_terminal': 0,
            'crk_c3_terminal': 0,
            'crk_c4_internal': 0,
            'crk_c4_terminal': 0,
            'crk_c5plus_internal': 0,
            'crk_c5plus_terminal': 0,
        }
        
        # Count sites for each chain
        for start, end in self.chains:
            chain_length = end - start
            chain_segment = self.carbon_array[start:end]
            
            # Count adsorption sites
            self._count_adsorption_sites(chain_segment, chain_length, counts)
            
            # Count desorption sites
            self._count_desorption_sites(chain_segment, chain_length, counts)
            
            # Count dehydrogenation (double M-C formation) sites
            self._count_dmc_sites(chain_segment, chain_length, counts)
            
            # Count cracking sites
            self._count_cracking_sites(chain_segment, chain_length, counts)
        
        return counts

    def _identify_chains(self):
        """
        Identify separate chains by checking connectivity in chain_array.
        Returns list of (start, end) tuples for each chain.
        """
        chains = []
        current_chain = [0]  # Start with the first carbon
        
        # Iterate through chain_array to find connections
        n = len(self.chain_array)
        for i in range(1, n):
            if self.chain_array[i] == 1:  # Connected to previous carbon
                current_chain.append(i)
            else:  # Not connected, start new chain
                # Save current chain
                start = current_chain[0]
                end = current_chain[-1] + 1
                chains.append((start, end))
                # Start new chain
                current_chain = [i]
        
        return chains

    def _count_adsorption_sites(self, chain_segment, chain_length, counts):
        """
        Count adsorption sites (0s in carbon_array) by chain length.
        Only count if the chain is NOT already adsorbed.
        """
        # Skip if any carbon in this chain is already adsorbed
        if np.any(chain_segment == 1):
            return
        
        num_zeros = np.sum(chain_segment == 0)
        
        if chain_length == 1:
            counts['ads_c1'] += num_zeros
        elif chain_length == 2:
            counts['ads_c2'] += num_zeros
        elif chain_length == 3:
            counts['ads_c3'] += num_zeros
        elif chain_length == 4:
            counts['ads_c4'] += num_zeros
        else:  # C5+
            # Count internal zeros
            if chain_length > 2:
                internal_zeros = np.sum(chain_segment[1:-1] == 0)
                counts['ads_c5plus_internal'] += internal_zeros
            # Count terminal zeros
            terminal_zeros = 0
            if chain_segment[0] == 0:
                terminal_zeros += 1
            if chain_segment[-1] == 0:
                terminal_zeros += 1
            counts['ads_c5plus_terminal'] += terminal_zeros

    def _count_desorption_sites(self, chain_segment, chain_length, counts):
        """
        Count desorption sites (1s in carbon_array) by chain length.
        Only count if exactly ONE carbon is adsorbed.
        """
        num_attached = np.sum(chain_segment == 1)
        
        # Only count if exactly one carbon is attached
        if num_attached != 1:
            return
        
        # Find which position is attached
        attached_idx = np.where(chain_segment == 1)[0][0]
        is_terminal = (attached_idx == 0) or (attached_idx == chain_length - 1)
        
        if chain_length == 1:
            counts['des_c1'] += 1
        elif chain_length == 2:
            counts['des_c2'] += 1
        elif chain_length == 3:
            counts['des_c3'] += 1
        elif chain_length == 4:
            counts['des_c4'] += 1
        else:  # C5+
            if is_terminal:
                counts['des_c5plus_terminal'] += 1
            else:
                counts['des_c5plus_internal'] += 1

    def _count_dmc_sites(self, chain_segment, chain_length, counts):
        """
        Count double M-C bond formation sites (dehydrogenation).
        Requires exactly one carbon adsorbed, and counts vacant neighbors.
        
        C1: No dMC (can't form double M-C with single carbon)
        C2, C3: Only terminal dMC
        C4+: Both internal and terminal dMC
        """
        num_attached = np.sum(chain_segment == 1)
        
        # Only count if exactly one carbon is attached
        if num_attached != 1:
            return
        
        # No dMC for C1
        if chain_length == 1:
            return
        
        # Find the attached carbon position
        attached_idx = np.where(chain_segment == 1)[0][0]
        
        # Count vacant neighbors and track which ones
        left_vacant = attached_idx > 0 and chain_segment[attached_idx - 1] == 0
        right_vacant = attached_idx < chain_length - 1 and chain_segment[attached_idx + 1] == 0
        vacant_neighbors = int(left_vacant) + int(right_vacant)
        
        if vacant_neighbors == 0:
            return
        
         # For C2 and C3, everything is terminal
        if chain_length == 2:
            counts['dmc_c2_terminal'] += vacant_neighbors
        elif chain_length == 3:
            counts['dmc_c3_terminal'] += vacant_neighbors
        else:  # C4+
            # Check if dMC formation would involve a terminal carbon
            # Terminal if: adsorbed carbon is at end, OR vacant neighbor is at end
            terminal_count = 0
            internal_count = 0
        
            if left_vacant:
                # Forms bond at position (attached_idx-1, attached_idx)
                bond_position = attached_idx - 1
                if bond_position == 0 or attached_idx == chain_length - 1:  # Bond involves terminal carbon
                    terminal_count += 1
                else:
                    internal_count += 1
            
            if right_vacant:
                # Forms bond at position (attached_idx, attached_idx+1)
                bond_position = attached_idx
                if bond_position == 0 or bond_position == chain_length - 2:  # Bond at either end
                    terminal_count += 1
                else:
                    internal_count += 1
            
            if chain_length == 4:
                counts['dmc_c4_terminal'] += terminal_count
                counts['dmc_c4_internal'] += internal_count
            else:  # C5+
                counts['dmc_c5plus_terminal'] += terminal_count
                counts['dmc_c5plus_internal'] += internal_count

    def _count_cracking_sites(self, chain_segment, chain_length, counts):
        """
        Count C-C bond scission sites.
        Requires pattern: 11 (two adjacent carbons adsorbed)
        
        C1: No cracking (no C-C bonds)
        C2, C3: Only terminal cracking (all C-C bonds are "terminal-like")
        C4+: Both internal and terminal cracking
        """
        # No cracking for C1
        if chain_length == 1:
            return
        
        # Look for 11 patterns (two adjacent adsorbed carbons)
        crack_sites = []
        for i in range(chain_length - 1):
            if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                crack_sites.append(i)
        
        if len(crack_sites) == 0:
            return
        
        # For C2 and C3, all cracking is "terminal"
        if chain_length == 2:
            counts['crk_c2_terminal'] += len(crack_sites)
        elif chain_length == 3:
            counts['crk_c3_terminal'] += len(crack_sites)
        elif chain_length == 4:
            # Distinguish internal vs terminal
            for site_idx in crack_sites:
                # Terminal: bond at position 0-1 or (n-2)-(n-1)
                is_terminal = (site_idx == 0) or (site_idx == chain_length - 2)
                if is_terminal:
                    counts['crk_c4_terminal'] += 1
                else:
                    counts['crk_c4_internal'] += 1
        else:  # C5+
            for site_idx in crack_sites:
                is_terminal = (site_idx == 0) or (site_idx == chain_length - 2)
                if is_terminal:
                    counts['crk_c5plus_terminal'] += 1
                else:
                    counts['crk_c5plus_internal'] += 1