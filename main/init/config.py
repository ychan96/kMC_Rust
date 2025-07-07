import numpy as np
from kmc.init import BaseKineticMC

class ConfigMixin:
    def update_configuration(self):
        #self.chains = self._identify_chains() - commmented out bcuz it's overriding the property
        counts = np.zeros(9, int)
        # Count available sites for each reaction type
        counts[0] = self.count_internal_adsorption_sites()   # Internal adsorption
        counts[1] = self.count_terminal_adsorption_sites() # Terminal adsorption
        counts[2] = self.count_internal_desorption_sites()   # Internal desorption
        counts[3] = self.count_terminal_desorption_sites()   # Terminal desorption
        counts[4] = self.count_internal_dehydrogenation_sites() # Internal dehydrogenation
        counts[5] = self.count_terminal_dehydrogenation_sites() # Terminal dehydrogenation
        counts[6] = self.count_doubleMC_desorption_sites()      # Desorption of double M-C bonds
        counts[7] = self.count_internal_cracking_sites()        # Internal cracking
        counts[8] = self.count_terminal_cracking_sites()        # Terminal cracking

        return counts

    def _identify_chains(self):
        
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
            
            # Add the last chain
            if current_chain:
                start = current_chain[0]
                end = current_chain[-1] + 1
                chains.append((start, end))
            
            return chains

    def count_internal_adsorption_sites(self):

            # Count internal unattached sites (carbon=0) in chains of length > 2
            count = 0
            for start, end in self.chains:
                chain_segment = self.carbon_array[start:end]
                # Only count internal sites in chains with 3+ carbons
                if len(chain_segment) > 2:
                    #if not np.any(chain_segment == 1):
                        internal_zeros = 0
                        for i in range(1, len(chain_segment) - 1):
                            if chain_segment[i] == 0:
                                internal_zeros += 1
                        count += internal_zeros
            
            return count
        
    def count_terminal_adsorption_sites(self):

        
        # Count terminal sites with no attachment
        count = 0
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 1: # and not np.any(chain_segment == 1):
                if chain_segment[0] == 0:
                    count += 1
                if len(chain_segment) > 1 and chain_segment[-1] == 0:
                    count += 1
        
        return count

    def count_internal_desorption_sites(self):

        # Count attached carbons that can desorb
        count = 0
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >=  3:
                num_attached = np.sum(chain_segment[1:-1] == 1) 
                if num_attached == 1:
                    count += 1
                    
        return count

    def count_terminal_desorption_sites(self):
        
        # Count attached carbons that can desorb
        count = 0
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 2:
                num_attached = np.sum(chain_segment == 1)
                if num_attached == 1:
                    if chain_segment[0] == 1:
                        count += 1
                    elif chain_segment[-1] == 1:
                        count += 1
            elif len(chain_segment) == 1:  # Single carbon chain
                if chain_segment[0] == 1:
                    count += 1
        
        return count

    def count_internal_dehydrogenation_sites(self):

        # Count sites where dehydrogenation can occur
        count = 0
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >= 4 and np.sum(chain_segment) == 1:
                # Look for patterns where a 1 is surrounded by 0s
                if chain_segment[1] == 1:
                    count += 1
                elif chain_segment[-2] == 1:
                    count += 1
                else: 
                    for i in range(2, len(chain_segment)-2):
                        if chain_segment[i] == 1 and chain_segment[i-1] == 0 and chain_segment[i+1] == 0:
                            count += 2  # Two adjacent sites can be dehydrogenated

        return count

    def count_terminal_dehydrogenation_sites(self):

        # Count terminal sites that can undergo dehydrogenation
        count = 0
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) == 2 and np.sum(chain_segment) == 1:
                if chain_segment[0] == 1:
                    count += 1
                elif chain_segment[1] == 1:
                    count += 1
            if len(chain_segment) == 3 and np.sum(chain_segment) == 1:                    
                if chain_segment[0] == 1:
                    count += 1
                elif chain_segment[1] == 1:
                    count += 2
                elif chain_segment[2] == 1:
                    count += 1
            if len(chain_segment) >= 4 and np.sum(chain_segment) == 1:
                if chain_segment[0] == 1:
                    count += 1
                elif chain_segment[1] == 1:
                    count += 1
                elif chain_segment[-1] == 1:
                    count += 1
                elif chain_segment[-2] == 1:
                    count += 1
        return count

    def count_doubleMC_desorption_sites(self):

        # Count attached carbons that can desorb
        count = 0
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            if len(chain_segment) >=  2:
                for i in range(len(chain_segment)-1):
                    if chain_segment[i] == 1 and chain_segment[i+1] == 1:
                        count += 2  # Count both carbons in the pair
                    
        return count

    def count_internal_cracking_sites(self):

        
        # Count internal sites for cracking (pattern 0110)
        count = 0
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            # Need at least 4 carbons for internal cracking
            if len(chain_segment) >= 4:
                for i in range(len(chain_segment)-3):
                    # Check for 0110 pattern
                    if (chain_segment[i] == 0 and 
                        chain_segment[i+1] == 1 and 
                        chain_segment[i+2] == 1 and 
                        chain_segment[i+3] == 0):
                        count += 1
        
        return count

    def count_terminal_cracking_sites(self):

        
        # Count terminal sites for cracking
        count = 0
        for start, end in self.chains:
            chain_segment = self.carbon_array[start:end]
            # Need at least 2 carbons for terminal cracking
            if len(chain_segment) >= 2:
                # Check for 110 pattern at start
                if (chain_segment[0] == 1 and 
                    chain_segment[1] == 1 ):
                    count += 1
                
                # Check for 011 pattern at end
                elif len(chain_segment) >= 2 and (
                    chain_segment[-2] == 1 and 
                    chain_segment[-1] == 1):
                    count += 1
        
        return count

    #monkey patching
    """ BaseKineticMC.update_configuration = update_configuration
        BaseKineticMC._identify_chains = _identify_chains
        BaseKineticMC.count_internal_adsorption_sites = count_internal_adsorption_sites
        BaseKineticMC.count_terminal_adsorption_sites = count_terminal_adsorption_sites
        BaseKineticMC.count_internal_desorption_sites = count_internal_desorption_sites
        BaseKineticMC.count_terminal_desorption_sites = count_terminal_desorption_sites
        BaseKineticMC.count_internal_dehydrogenation_sites = count_internal_dehydrogenation_sites
        BaseKineticMC.count_terminal_dehydrogenation_sites = count_terminal_dehydrogenation_sites
        BaseKineticMC.count_doubleMC_desorption_sites = count_doubleMC_desorption_sites
        BaseKineticMC.count_internal_cracking_sites = count_internal_cracking_sites
        BaseKineticMC.count_terminal_cracking_sites = count_terminal_cracking_sites """