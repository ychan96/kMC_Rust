import numpy as np
import random

class CoverageMixin:
    """
    Tracks metal surface state using two separate matrices:
    
    m_bond:  bond type at each site
             0 = vacant
             1 = single M-C, internal carbon
             2 = single M-C, terminal carbon
             3 = double M-C, internal carbon
             4 = double M-C, terminal carbon

    m_chain: chain length at each site
             +N = first carbon of chain N (single M-C)
             -N = second carbon of same chain N (double M-C partner)
              0 = vacant
    
    Example: C7 internal adsorption → double M-C formation
        m_bond:  [..., 1, ...] → [..., 1, 3, ...]  (single → double internal)
        m_chain: [..., 7, ...] → [..., -7, -7, ...]
    """

    def metal_surface(self, reaction_key):
        if reaction_key.startswith('ads_'):
            return self.coverage_adsorption(reaction_key)
        elif reaction_key.startswith('des_'):
            return self.coverage_desorption(reaction_key)
        elif reaction_key.startswith('dmc_'):
            return self.coverage_dmc_formation(reaction_key)
        elif reaction_key.startswith('crk_'):
            return self.coverage_cracking(reaction_key)
        return False

    def update_theta(self):
        total_sites = self.m_size * self.m_size
        occupied_sites = np.count_nonzero(self.m_bond)
        self.theta = occupied_sites / total_sites
        return self.theta

    def _get_vacant_sites(self):
        return np.argwhere(self.m_bond == 0)

    def _get_neighbor_vacant(self, i, j):
        """Return vacant neighbor sites of (i,j) as list of (row, col)"""
        rows, cols = self.m_bond.shape
        row_start = max(0, i - 1)
        row_end   = min(rows, i + 2)
        col_start = max(0, j - 1)
        col_end   = min(cols, j + 2)
        #extract submatrix (for neighbors)
        sub = self.m_bond[row_start:row_end, col_start:col_end]
        local_vacants = np.argwhere(sub == 0)
        # Convert to global coordinates
        return [(row_start + r, col_start + c) for r, c in local_vacants]

    def _get_neighbor_by_bond(self, i, j, bond_type):
        """
        Return neighboring sites matching bond_type as list of (row, col)
        e.g., partners = self._get_neighbor_by_bond(i, j, bond_type=3)
            -> returns list; [(n_i, n_j), ...] of neighbors with double M-C internal bond
        """
        rows, cols = self.m_bond.shape
        row_start = max(0, i - 1)
        row_end   = min(rows, i + 2)
        col_start = max(0, j - 1)
        col_end   = min(cols, j + 2)
        sub = self.m_bond[row_start:row_end, col_start:col_end]
        local_matches = np.argwhere(sub == bond_type) 
        # bond_type can be: 
        # 1 = single M-C, internal carbon
        # 2 = single M-C, terminal carbon
        # 3 = double M-C, internal carbon
        # 4 = double M-C, terminal carbon
        return [(row_start + r, col_start + c) for r, c in local_matches]
    
    def _get_neighbor_by_chain(self, i, j, chain_value):
        """Return neighboring sites with specific m_chain value"""
        rows, cols = self.m_chain.shape
        row_start = max(0, i - 1)
        row_end   = min(rows, i + 2)
        col_start = max(0, j - 1)
        col_end   = min(cols, j + 2)
        
        neighbors = []
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                if (r, c) != (i, j) and self.m_chain[r, c] == chain_value:
                    neighbors.append((r, c))
        return neighbors
    
    def _get_bond_type_from_adsorption(self, chain_len, ads_key):
        """Check carbon_array to see if adsorption was internal or terminal"""
        # For C1,C2, always terminal
        if chain_len <= 2:
            return 2
        
        # For C4+, check the carbon_array
        for start, end in self.chains:
            if end - start != chain_len:
                continue
            chain_segment = self.carbon_array[start:end]
            
            # Find where the 1 is
            ones = np.where(chain_segment == 1)[0]
            if len(ones) == 1:
                pos = ones[0]
                is_terminal = (pos == 0) or (pos == chain_len - 1)
                return 2 if is_terminal else 1
        
        # Fallback
        if 'internal' in ads_key:
            return 1
        return 2
    
    def _get_fragment_lengths(self, original_len, crk_key):
        """
        Determine fragment lengths after C-C cleavage.
        Uses the carbon_array cracking position for accurate split.
        """
        for start, end in self.chains:
            chain_length = end - start
            if chain_length != original_len:
                continue
            chain_segment = self.carbon_array[start:end]

            # Find the 11 pattern (double M-C bond position)
            for i in range(chain_length - 1):
                if chain_segment[i] == 1 and chain_segment[i + 1] == 1:
                    frag1 = i + 1           # Carbons 0..i
                    frag2 = chain_length - frag1  # Carbons i+1..end
                    return frag1, frag2

        # Fallback: split roughly in half
        frag1 = original_len // 2
        frag2 = original_len - frag1
        return frag1, frag2

    def _get_reacting_chain_length(self, ads_key):
        """
        For C5+ adsorption, get actual chain length from carbon_array.
        """
        is_internal = 'internal' in ads_key

        for start, end in self.chains:
            chain_length = end - start
            if chain_length < 5:
                continue
            chain_segment = self.carbon_array[start:end]
            if np.any(chain_segment == 1) == 1:
            # Return first matching chain length
                return chain_length

        return 5  # Fallback
    
    def coverage_adsorption(self, ads_key):
        # Determine chain length
        if 'c1' in ads_key:
            chain_len = 1
        elif 'c2' in ads_key:
            chain_len = 2
        elif 'c3' in ads_key:
            chain_len = 3
        elif 'c4' in ads_key:
            chain_len = 4
        else:  # c5plus
            chain_len = self._get_reacting_chain_length(ads_key)
        
        # Determine if the actual adsorption was internal or terminal
        # by looking at where the 1 appeared in carbon_array
        bond_type = self._get_bond_type_from_adsorption(chain_len, ads_key)
        
        vacant_sites = self._get_vacant_sites()
        if len(vacant_sites) == 0:
            return False
        
        row, col = random.choice(vacant_sites)
        #assign bond type(1,2,3,4) and chain length
        self.m_bond[row, col]  = bond_type
        self.m_chain[row, col] = chain_len
        
        self.update_theta()
        return True

    def coverage_desorption(self, des_key):
        """
        Remove a single M-C bond carbon from the surface.
        For C1-C4: checks actual bond_type in m_bond matrix
        For C5+: uses key to determine internal/terminal
        """
        # Determine target chain length range
        if 'c1' in des_key:
            target_len = 1
        elif 'c2' in des_key:
            target_len = 2
        elif 'c3' in des_key:
            target_len = 3
        elif 'c4' in des_key:
            target_len = 4
        else:
            target_len = None  # C5+

        # Find matching sites
        matching_sites = []
        
        # For C1-C4: check both bond_type 1 and 2
        if target_len is not None:
            for bond_type in [1, 2]:
                candidate_sites = np.argwhere(self.m_bond == bond_type)
                # Final check for chain length match
                for i, j in candidate_sites:
                    if self.m_chain[i, j] == target_len:
                        matching_sites.append((i, j))
        else:  # C5+: use key to determine bond_type
            is_internal = 'internal' in des_key
            bond_type = 1 if is_internal else 2
            candidate_sites = np.argwhere(self.m_bond == bond_type)
            # Final check for chain length >=5
            for i, j in candidate_sites:
                if self.m_chain[i, j] >= 5:
                    matching_sites.append((i, j))

        if not matching_sites:
            return False

        i, j = random.choice(matching_sites)
        self.m_bond[i, j]  = 0
        self.m_chain[i, j] = 0

        self.update_theta()
        return True
    
    def coverage_dmc_formation(self, dmc_key):
        """
        Form double M-C bond by placing second carbon on neighboring vacant site.

        Before: m_bond[i,j] = 1 or 2,  m_chain[i,j] = +N
        After:  m_bond[i,j] unchanged,  new neighbor gets bond_type 3 or 4
                m_chain[i,j] = -N & m_chain[neighbor] = -N  (negative = double bond partner)
        """
        # Determine which chain length we're looking for
        if 'c2' in dmc_key:
            target_len = 2
        elif 'c3' in dmc_key:
            target_len = 3
        elif 'c4' in dmc_key:
            target_len = 4
        else:  # c5plus
            target_len = None  # Any length >= 5
        
        # Find all sites with single bonds (type 1 or 2) matching chain length
        eligible = []
        
        for bond_type in [1, 2]:  # Check both internal and terminal
            candidate_sites = np.argwhere(self.m_bond == bond_type)
            
            for i, j in candidate_sites:
                chain_len = self.m_chain[i, j]
                
                # Check if chain length matches
                if target_len is not None:
                    if chain_len != target_len:
                        continue
                else:  # c5plus
                    if chain_len < 5:
                        continue
                    # For c5plus, also check internal/terminal from key
                    is_internal = 'internal' in dmc_key
                    if (bond_type == 1) != is_internal:
                        continue
                
                # Check for vacant neighbors
                vacants = self._get_neighbor_vacant(i, j)
                if vacants:
                    eligible.append((i, j, vacants, bond_type))
        
        if not eligible:
            return False
        
        # Pick a random eligible site
        i, j, vacants, source_bond = random.choice(eligible)
        chain_len = self.m_chain[i, j]
        
        # Determine new bond type based on source
        new_bond = 3 if source_bond == 1 else 4  # internal→3, terminal→4
        
        # Place second carbon on a random vacant neighbor
        ni, nj = random.choice(vacants)
        self.m_bond[ni, nj]  = new_bond
        self.m_chain[i, j]   = -chain_len  # Source becomes negative
        self.m_chain[ni, nj] = -chain_len  # Partner is negative
        
        self.update_theta()
        return True

    def coverage_cracking(self, crk_key):
        """
        C-C bond cleavage splits chain N into two fragments.

        Before cracking (internal C7 example):
            m_bond:  [..., 1, 3, ...]   m_chain: [..., -7, -7, ...]

        After cracking into C3 + C4 (internal):
            m_bond:  [..., 2, 2, ...]   m_chain: [..., 3, 4, ...]
            (both become terminal carbons of their new chains)

        After cracking (terminal C7 example → C1 + C6):
            m_bond:  [..., 2, 2, ...]   m_chain: [..., 1, 6, ...]
        """
        # Determine which chain length we're looking for
        if 'c2' in crk_key:
            target_len = 2
        elif 'c3' in crk_key:
            target_len = 3
        elif 'c4' in crk_key:
            target_len = 4
        else:  # c5plus
            target_len = None  # Any length >= 5

        # Find pairs of double-bonded carbons (both have negative chain lengths)
        # Look for adjacent sites where both m_chain values are negative and equal
        eligible = []

        all_negative_sites = np.argwhere(self.m_chain < 0)
    
        for i, j in all_negative_sites:
            chain_len = abs(self.m_chain[i, j])
            
            # Check if chain length matches
            if target_len is not None:
                if chain_len != target_len:
                    continue
            else:  # c5plus
                if chain_len < 5:
                    continue
                # For c5plus, check internal/terminal from key
                is_internal = 'internal' in crk_key
                bond_type = self.m_bond[i, j]

                # Internal: bond_type 1 or 3; Terminal: bond_type 2 or 4 
                is_site_internal = (bond_type == 1 or bond_type == 3)
                if is_site_internal != is_internal:
                    continue
            
            # Find the partner (adjacent site with same negative chain length)
            neighbors = self._get_neighbor_by_chain(i, j, -chain_len)
            if neighbors:
                # Only add once (avoid duplicates)
                partner = neighbors[0]
                pair = tuple(sorted([(i, j), partner])) # sort to avoid duplicates
                if pair not in [e[0] for e in eligible]:
                    eligible.append((pair, chain_len))
                """
                eligible = [
                    (((1, 2), (1, 3)), 7),  # e[0] = ((1, 2), (1, 3))
                    (((2, 1), (2, 2)), 5),  # e[0] = ((2, 1), (2, 2))
                ]

                [e[0] for e in eligible]
                # Returns: [((1, 2), (1, 3)), ((2, 1), (2, 2))]

                # Then checks:
                if new_pair not in [((1, 2), (1, 3)), ((2, 1), (2, 2))]:
                    # Add to eligible
                """
        
        if not eligible:
            return False
        
        # Pick a random eligible pair
        pair, original_len = random.choice(eligible)
        (i1, j1), (i2, j2) = pair
        
        # Determine fragment lengths from carbon_array
        frag1, frag2 = self._get_fragment_lengths(original_len, crk_key)
        
        # Update both sites: both become terminal (bond_type = 2)
        self.m_bond[i1, j1]  = 2
        self.m_chain[i1, j1] = frag1
        
        self.m_bond[i2, j2]  = 2
        self.m_chain[i2, j2] = frag2
        
        self.update_theta()
        return True