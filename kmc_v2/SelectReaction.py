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

    Arrays
        carbon_array[i]: i-th carbon = 0 or 1
        hydrogen_array[i]: # of H atoms on i-th carbon 3 or 2

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
        
    def sample_adsorption_site(self, free_positions, chain_start, chain_length, use_normal=True):
        if not free_positions:
            return None

        if use_normal:
            mid     = chain_start + (chain_length - 1) / 2
            sigma   = chain_length / 8
            weights = stats.norm.pdf(free_positions, loc=mid, scale=sigma)
            weights /= weights.sum()
            return int(np.random.choice(free_positions, p=weights))
        else:
            return int(np.random.choice(free_positions))
    
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
        free_positions = list(np.where(seg == 0)[0] + start) #global position
        sampled = self.sample_adsorption_site(free_positions, start, N, use_normal=True)
        local_c      = np.random.choice(sampled) - start
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
        chosen_h_sites = np.random.choice(vacant_h_sites, size=n_h_released, replace=False) #no duplicates
        self.h_occupancy[chosen_h_sites] = 1
        
        return True
    
    def perform_desorption(self, N):
        # 1. Find all single-MC carbons in length-N fragments
        candidate_carbons = []
        for start, end in self.chains:
            if end - start != N: #chain-length matching
                continue
            seg = self.carbon_array[start:end]
            if int(np.sum(seg == 1)) != 1: #is it adsorbed??
                continue
            local_pos  = np.where(seg == 1)[0][0] #accesses the first index array and first index
            global_c   = start + local_pos
            site_idx   = self.carbon_to_site[global_c]
            if site_idx == -1 or self.occupancy[site_idx] != 1:
                continue
            candidate_carbons.append((global_c, local_pos, N))

        if not candidate_carbons:
            return False

        # 2. Pick one randomly
        global_c, local_pos, chain_len = candidate_carbons[np.random.choice(len(candidate_carbons))]
        is_terminal = (local_pos == 0) or (local_pos == chain_len - 1)
        site_idx    = self.carbon_to_site[global_c] #find surface site

        # 3. Update carbon arrays
        self.carbon_array[global_c]   = 0
        self.hydrogen_array[global_c] = 3 if is_terminal else 2  # restore H count

        # 4. Update C site arrays
        self.occupancy[site_idx]      = 0
        self.carbon_at_site[site_idx] = -1
        self.chain_at_site[site_idx]  = 0
        self.carbon_to_site[global_c] = -1

        # 5. Return H atoms — free random hollow sites
        n_h_returned    = 3 if is_terminal else 2
        occupied_h      = np.where(self.h_occupancy == 1)[0]
        chosen_h_sites  = np.random.choice(occupied_h, size=n_h_returned, replace=False)
        self.h_occupancy[chosen_h_sites] = 0

        return True
    
    def perform_dmc_formation(self, N, pos):
        # 1. Find eligible fragments — exactly one single-MC carbon with a free neighbor
        candidate_pairs = []
        for start, end in self.chains:
            if end - start != N:
                continue
            seg = self.carbon_array[start:end]
            if int(np.sum(seg == 1)) != 1:
                continue

            idx      = np.where(seg == 1)[0][0]
            site_idx = self.carbon_to_site[start + idx]
            if site_idx == -1 or self.occupancy[site_idx] != 1:
                continue

            # Check vacant surface neighbor exists
            if not any(self.occupancy[nb] == 0 for nb in self.surface.get_c_neighbors(site_idx)):
                continue

            for nb_pos in [idx - 1, idx + 1]:
                if nb_pos < 0 or nb_pos >= N:
                    continue
                if seg[nb_pos] != 0:
                    continue
                at_terminal = (idx == 0 or idx == N-1 or nb_pos == 0 or nb_pos == N-1)
                if (pos == 'internal') == (not at_terminal):
                    candidate_pairs.append((start + idx, start + nb_pos))

        if not candidate_pairs:
            return False

        # 2. Pick one pair
        anchor_c, new_c = candidate_pairs[np.random.choice(len(candidate_pairs))]
        anchor_site     = self.carbon_to_site[anchor_c]

        # 3. Find vacant surface neighbor for new_c
        vacant_nb = [nb for nb in self.surface.get_c_neighbors(anchor_site)
                    if self.occupancy[nb] == 0]
        new_site  = np.random.choice(vacant_nb)

        # 4. Update carbon arrays
        self.carbon_array[new_c]   = 1
        self.hydrogen_array[new_c] = 0

        # 5. Update C site arrays
        self.occupancy[anchor_site]      = 2        # upgrade anchor to dMC
        self.occupancy[new_site]         = 2
        self.carbon_at_site[new_site]    = new_c
        self.chain_at_site[new_site]     = N
        self.carbon_to_site[new_c]       = new_site

        # 6. Release H atoms
        n_h_released   = 3 if (new_c == 0 or new_c == N - 1) else 2
        vacant_h       = np.where(self.h_occupancy == 0)[0]
        chosen_h       = np.random.choice(vacant_h, size=n_h_released, replace=False)
        self.h_occupancy[chosen_h] = 1

        return True


    def perform_cracking(self, N, pos):
        # 1. Find eligible 11 patterns
        candidate_bonds = []
        for start, end in self.chains:
            if end - start != N:
                continue
            seg = self.carbon_array[start:end]
            if int(np.sum(seg == 1)) != 2:
                continue

            for i in range(N - 1):
                if seg[i] == 1 and seg[i + 1] == 1:
                    at_terminal = (i == 0) or (i == N - 2)
                    if (pos == 'internal') == (not at_terminal):
                        candidate_bonds.append((start, end, start + i + 1))  # chain_array index

        if not candidate_bonds:
            return False

        # 2. Pick one bond
        start, end, chain_idx = candidate_bonds[np.random.choice(len(candidate_bonds))]
        seg       = self.carbon_array[start:end]
        local_i   = chain_idx - start - 1          # local position of left carbon

        # 3. Compute fragments
        frag1 = local_i + 1
        frag2 = N - frag1

        # 4. Get the two carbons
        g_left  = start + local_i
        g_right = start + local_i + 1
        site_l  = self.carbon_to_site[g_left]
        site_r  = self.carbon_to_site[g_right]

        # 5. Update carbon arrays — both carbons desorb to free state
        for g_c, site_idx in [(g_left, site_l), (g_right, site_r)]:
            local_pos = g_c - start
            is_terminal = (local_pos == 0) or (local_pos == N - 1)
            self.carbon_array[g_c]        = 0
            self.hydrogen_array[g_c]      = 3 if is_terminal else 2
            self.occupancy[site_idx]      = 0
            self.carbon_at_site[site_idx] = -1
            self.chain_at_site[site_idx]  = 0
            self.carbon_to_site[g_c]      = -1

        # 6. Break chain bond
        self.chain_array[chain_idx] = 0

        # 7. Return H atoms
        occupied_h = np.where(self.h_occupancy == 1)[0]
        np.random.choice(occupied_h, size=4, replace=False)  # 2+2 H returned
        self.h_occupancy[np.random.choice(occupied_h, size=4, replace=False)] = 0

        # 8. Update hydrogen_array for newly exposed terminals at break point
        self.hydrogen_array[g_left]  = 3   # becomes new terminal of left fragment
        self.hydrogen_array[g_right] = 3   # becomes new terminal of right fragment

        self.invalidate_chains()
        return True
    