import numpy as np, random
from kmc.init import BaseKineticMC

class ReactionMixin:
    def select_reaction(self, site_counts):

        #calculate rates
        rates = self.k_const * site_counts
        R = rates.sum() #Total rate 

        # no rxns
        if R == 0: return None, 0

        #generate random number
        u1 = np.random.rand()
        cum = np.cumsum(rates)/R #normalization

        idx = np.argmax(cum > u1) # reaction type
        u2 = np.random.rand() #independent to u1
        dt  = -np.log(u2)/R # time increment 
        return idx, dt

    def perform_reaction(self, idx):
        if idx == 0:  # Internal adsorption
            return self.perform_internal_adsorption()
        elif idx == 1:  # Terminal adsorption
            return self.perform_terminal_adsorption()
        elif idx == 2:  # Internal desorption
            return self.perform_internal_desorption()
        elif idx == 3:  # Terminal desorption
            return self.perform_terminal_desorption()
        elif idx == 4:  # Internal dehydrogenation
            return self.perform_internal_dehydrogenation()
        elif idx == 5:  # Terminal dehydrogenation
            return self.perform_terminal_dehydrogenation()
        elif idx == 6:  # DoubleM-C desorption
            return self.perform_doubleMC_desorption()
        elif idx == 7:  # Internal cracking
            return self.perform_internal_cracking()
        elif idx == 8:  # Terminal cracking
            return self.perform_terminal_cracking()
        return False

    def perform_internal_adsorption(self):

            # Find all possible adsorption sites
            adsorption_sites = []
            for start, end in self.chains:
                chain_segment = self.carbon_array[start:end]
                if len(chain_segment) >= 3:
                    if not np.any(chain_segment == 1):
                        for i in range(1, len(chain_segment)-1):
                            if chain_segment[i] == 0:
                                adsorption_sites.append(start + i)
            
            if not adsorption_sites:
                return False
            
            # Choose a random site and perform adsorption
            site = random.choice(adsorption_sites)
            self.carbon_array[site] = 1
            
            return True
        
    def perform_terminal_adsorption(self):
        
        # Find all possible terminal adsorption sites
        adsorption_sites = []
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 1:
                if not np.any(chain_segment == 1):
                    if chain_segment[0] == 0:
                        adsorption_sites.append(start)
                    elif len(chain_segment) > 1 and chain_segment[-1] == 0:
                        adsorption_sites.append(end - 1)
        
        if not adsorption_sites:
            return False
        
        # Choose a random site and perform adsorption
        site = random.choice(adsorption_sites)
        self.carbon_array[site] = 1
        
        return True

    def perform_internal_desorption(self):
        
        # Find all possible desorption sites
        desorption_sites = []
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 3:

                has_adjacent = False
                for i in range(len(chain_segment)-1):
                    if chain_segment[i] == 1 and chain_segment[i+1] == 1:
                        has_adjacent = True
                        break
                    
                if not has_adjacent:
                    # Find single 1s in internal positions
                    for i in range(1, len(chain_segment)-1):
                        if chain_segment[i] == 1:
                            desorption_sites.append(start+i)
        if not desorption_sites:
            return False
        
        # Choose a random site and perform desorption
        site = random.choice(desorption_sites)
        self.carbon_array[site] = 0
        
        return True

    def perform_terminal_desorption(self):

        # Find all possible desorption sites
        desorption_sites = []
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 2:

                has_adjacent = False
                for i in range(len(chain_segment)-1):
                    if chain_segment[i] == 1 and chain_segment[i+1] == 1:
                        has_adjacent = True
                        break

                if not has_adjacent:
                    if chain_segment[0] == 1:
                        desorption_sites.append(start)
                    elif chain_segment[-1] == 1:
                        desorption_sites.append(end - 1)

            elif len(chain_segment) == 1:
                if chain_segment[0] == 1:
                    desorption_sites.append(start)
                    
        if not desorption_sites:
            return False
        
        # Choose a random site and perform desorption
        site = random.choice(desorption_sites)
        self.carbon_array[site] = 0
        
        return True

    def perform_internal_dehydrogenation(self):
        
        # Find all possible dehydrogenation sites
        dehydrogenation_sites = []
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 4 and np.sum(chain_segment) == 1:
                if chain_segment[1] == 1:
                    dehydrogenation_sites.append(start+1)
                elif chain_segment[-2] == 1:
                    dehydrogenation_sites.append(end-2)
                else: 
                    for i in range(2, len(chain_segment)-2):
                        if chain_segment[i] == 1 and chain_segment[i-1] == 0 and chain_segment[i+1] == 0:
                            dehydrogenation_sites.append(start+i)

        
        if not dehydrogenation_sites:
            return False
        
        # Choose a random site and perform dehydrogenation of adjacent site
        site = random.choice(dehydrogenation_sites)
        adjacent_sites = []
        if self.carbon_array[site-1] == 0 and self.chain_array[site-1] == 0: #0100... case
            adjacent_sites.append(site+1)
        elif self.carbon_array[site+1] == 0 and self.chain_array[site+2] == 0: #...0010
            adjacent_sites.append(site-1)
        else:
            adjacent_sites.extend([site+1, site-1]) 
            
        if not adjacent_sites:
            return False
            
        # Pick random adjacent site and perform dehydrogenation
        adj_site = random.choice(adjacent_sites)
        self.carbon_array[adj_site] = 1
        
        return True

    def perform_terminal_dehydrogenation(self):
        
        # Find all possible terminal dehydrogenation sites
        dehydrogenation_sites = []
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) == 2 and np.sum(chain_segment) == 1:
                if chain_segment[0] == 1:
                    dehydrogenation_sites.append(start + 1)
                elif chain_segment[1] == 1:
                    dehydrogenation_sites.append(start)
            elif len(chain_segment) == 3 and np.sum(chain_segment) == 1:
                if chain_segment[0] == 1:
                    dehydrogenation_sites.append(start + 1)
                elif chain_segment[1] == 1:
                    dehydrogenation_sites.extend([start, start+2])
                elif chain_segment[2] == 1:
                    dehydrogenation_sites.append(start + 1)
            elif len(chain_segment) >= 4 and np.sum(chain_segment) == 1:
                if chain_segment[0] == 1:
                    dehydrogenation_sites.append(start + 1)
                elif chain_segment[1] == 1:
                    dehydrogenation_sites.append(start)
                elif chain_segment[-1] == 1:
                    # Make sure we don't exceed the array bounds
                    if end - 2 < len(self.carbon_array):
                        dehydrogenation_sites.append(end - 2)
                elif chain_segment[-2] == 1:
                    # Make sure we don't exceed the array bounds
                    if end - 1 < len(self.carbon_array):
                        dehydrogenation_sites.append(end - 1)
        
        if not dehydrogenation_sites:
            return False
        
        # Choose a random site and perform dehydrogenation
        site = random.choice(dehydrogenation_sites)
        
        # Make sure the site is within bounds
        if 0 <= site < len(self.carbon_array):
            self.carbon_array[site] = 1
            return True
        
        return False

    def perform_doubleMC_desorption(self):

        # Find all possible desorption sites
        desorption_sites = []
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 2:
                for i in range(len(chain_segment)-1):
                    if i+1 < len(chain_segment) and chain_segment[i] == 1 and chain_segment[i+1] == 1:
                        desorption_sites.extend([start+i, start+i+1])

        if not desorption_sites:
            return False
        
        # Choose a random site and perform desorption
        site = random.choice(desorption_sites)
        self.carbon_array[site] = 0
        
        return True

    def perform_internal_cracking(self):

        # Parameter to control how strongly we prefer middle positions
        # Higher B means more concentration around the middle
        #B = 0.8  # Can be tuned to adjust the distribution widt
        
        # Find all possible cracking sites (0110 pattern)
        cracking_sites = []
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 4:
                for i in range(len(chain_segment)-3):
                    if (chain_segment[i] == 0 and 
                        chain_segment[i+1] == 1 and 
                        chain_segment[i+2] == 1 and 
                        chain_segment[i+3] == 0):
                        # Store the position of the first "1" in the pattern
                        cracking_sites.append(start+i+2)
        
        if not cracking_sites:
            return False

        # Choose a random cracking site
        chain_index = random.choice(cracking_sites)
        
     
        #viable_chains = []
        
        #for start, end in self.chains:
        #    chain_length = end - start
            # Only consider chains of sufficient length for internal cracking
            # Need at least 4 carbons for meaningful internal cracking
        #    if chain_length >= 6:  # Minimum length for internal cracking
                # Store the chain range and its length
        #        viable_chains.append((start, end, chain_length))
        
        #if not viable_chains:
        #    return False
        
        # Choose a chain, with probability proportional to length
        # (longer chains more likely to crack)
        #chain_weights = [length for _, _, length in viable_chains]
        #selected_chain_idx = random.choices(
        #    range(len(viable_chains)), 
        #    weights=chain_weights, 
        #    k=1
        #)[0]
        
        #start, end, chain_length = viable_chains[selected_chain_idx]
        
        # For the selected chain, compute cracking probabilities for each position
        # We'll only consider internal positions (not the first or last bond)
        #positions = range(start + 1, end - 1)
        
        # Find the middle position
        #x_mid = (start + end - 1) / 2
        
        # Calculate the Gaussian probability for each position
        # P(x) = e^(-B*(x-x_mid)^2)
        #probabilities = []
        #for pos in positions:
        #    p = np.exp(-B * (pos - x_mid)**2)
        #    probabilities.append(p)
        
        # Normalize the probabilities
        #if sum(probabilities) > 0:
        #    probabilities = [p / sum(probabilities) for p in probabilities]
        #else:
            # Fallback to uniform probabilities if calculation fails
        #    probabilities = [1.0 / len(positions)] * len(positions)
        
        # Choose a cracking position based on the calculated probabilities
        #if positions and probabilities:
        #    chain_index = random.choices(positions, weights=probabilities, k=1)[0]
        
        # Break the chain by setting chain_array to 0
        self.chain_array[chain_index] = 0

        # Invalidate chains so they'll update new chain array
        self.invalidate_chains()
        
        return True

    def perform_terminal_cracking(self):
        
        # Find all possible terminal cracking sites
        cracking_sites = []
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 2:
                # Check for 11 pattern at start
                if (chain_segment[0] == 1 and 
                    chain_segment[1] == 1 ):
                    # Store the position to break the chain
                    cracking_sites.append(start+1)
                
                # Check for 11 pattern at end
                elif len(chain_segment) >= 2 and ( 
                    chain_segment[-2] == 1 and 
                    chain_segment[-1] == 1):
                    # Store the position to break the chain
                    cracking_sites.append(end-1)
        
        if not cracking_sites:
            return False
        
        # Choose a random cracking site and break the chain
        chain_index = random.choice(cracking_sites)
        self.chain_array[chain_index] = 0

        # Invalidate chains so they'll update new chain array
        self.invalidate_chains()
        
        return True

    #forget about monkey patching 
    """ BaseKineticMC.select_reaction  = select_reaction
        BaseKineticMC.perform_reaction = perform_reaction
        BaseKineticMC.perform_internal_adsorption = perform_internal_adsorption
        BaseKineticMC.perform_terminal_adsorption = perform_terminal_adsorption
        BaseKineticMC.perform_internal_desorption = perform_internal_desorption
        BaseKineticMC.perform_terminal_desorption = perform_terminal_desorption
        BaseKineticMC.perform_internal_dehydrogenation = perform_internal_dehydrogenation
        BaseKineticMC.perform_terminal_dehydrogenation = perform_terminal_dehydrogenation
        BaseKineticMC.perform_doubleMC_desorption = perform_doubleMC_desorption
        BaseKineticMC.perform_internal_cracking = perform_internal_cracking
        BaseKineticMC.perform_terminal_cracking = perform_terminal_cracking """