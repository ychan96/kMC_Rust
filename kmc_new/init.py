import numpy as np
from scipy import constants

eV = 1.60218e-19

class BaseKineticMC:
    def __init__(self, temp_C=250, reaction_time=7200, chain_length=None, 
                 rate_constants=None, P_H2=50, m_size=5): #desorption_penalty=0.0):
        # Parameters
        self.temp_K = temp_C + 273.15
        self.kb_eV = constants.Boltzmann / constants.e
        self.reaction_time = reaction_time
        self.P_H2 = P_H2  # FIX 1: Store P_H2 (you're accepting it but not storing it)
        
        # FIX 2: Initialize m_size and active sites BEFORE init_arrays
        # because init_arrays might need to know the surface geometry
        self.m_size = m_size
        #self.b = desorption_penalty
        self.init_active_sites(m_size)
        
        # Initialize arrays
        if chain_length is None:
            chain_length = self.normal_dist(mu=280, sigma=10, n_samples=None)
        
        self.chain_length = chain_length
        self.init_arrays(chain_length)
        
        # Initialize 16-parameter rate constant system
        if rate_constants is not None:
            self.k_const = rate_constants
        else:
            self.k_const = {
                # C1 (2 params)
                'ads_c1': 8.73535610614617e-03, #DEAD PARAMETER
                'des_c1': 5.23e-03,
                
                # C2 (2 params)
                'ads_c2': 3.8383383144364254e-04,
                'des_c2': 1.0e+00,
                
                # C3 (2 params)
                'ads_c3': 4.035893309943397e-02,
                'des_c3': 1.4315533590768602e-10,
                
                # C4 (2 params)
                'ads_c4': 1.0e+05,
                'des_c4': 2.18338647469809e-03,
                
                # C5+ (4 params)
                'ads_c5plus_internal': 1.60903758996922e-04,
                'des_c5plus_internal': 1.1426054526936904e-04,
                'ads_c5plus_terminal': 2.62750604280885e-03,
                'des_c5plus_terminal': 4.79374738412416e-03,
                
                # Double M-C (2 params)
                'dmc_terminal': 1.08998127942225e-03, #source of C1
                'dmc_internal': 1.5e-02,
                
                # Cracking (2 params)
                'crk_terminal': 1.5e-03,
                'crk_internal': 1.0e-03,
            }
        
        self.current_time = 0.0
        self._chains = None
        self._chains_valid = False
    
    @property  # FIX 3: Move @property decorator before the method definition
    def chains(self):
        """Lazy evaluation of chain identification"""
        if not self._chains_valid or self._chains is None: #when self._chains_valid is False -> clear cache
            self._chains = self._identify_chains()
            self._chains_valid = True
        return self._chains
    
    # Helper methods for rate lookup
    def get_adsorption_rate(self, chain_length, is_terminal):
        """Get adsorption rate based on chain length and position"""
        if chain_length == 1:
            return self.k_const['ads_c1']
        elif chain_length == 2:
            return self.k_const['ads_c2']
        elif chain_length == 3:
            return self.k_const['ads_c3']
        elif chain_length == 4:
            return self.k_const['ads_c4']
        else:  # C5+
            return self.k_const['ads_c5plus_terminal'] if is_terminal else self.k_const['ads_c5plus_internal']
    
    def get_desorption_rate(self, chain_length, is_terminal):
        """Get desorption rate based on chain length and position"""
        if chain_length == 1:
            return self.k_const['des_c1']
        elif chain_length == 2:
            return self.k_const['des_c2']
        elif chain_length == 3:
            return self.k_const['des_c3']
        elif chain_length == 4:
            return self.k_const['des_c4']
        else:  # C5+
            return self.k_const['des_c5plus_terminal'] if is_terminal else self.k_const['des_c5plus_internal']
    
    def get_dmc_rate(self, chain_length, is_terminal):
        """Get double M-C bond formation rate (dehydrogenation)
        Returns None for C1 (no dMC reaction)
        """
        if chain_length == 1:
            return None
        elif chain_length in [2, 3]:
            return self.k_const['dmc_terminal']
        else:  # C4+
            return self.k_const['dmc_terminal'] if is_terminal else self.k_const['dmc_internal']
    
    def get_cracking_rate(self, chain_length, is_terminal):
        """Get C-C scission rate
        Returns None for C1 (no C-C bonds)
        """
        if chain_length == 1:
            return None
        elif chain_length in [2, 3]:
            return self.k_const['crk_terminal']
        else:  # C4+
            return self.k_const['crk_terminal'] if is_terminal else self.k_const['crk_internal']
    
    def normal_dist(self, mu=260, sigma=30, n_samples=None):
        """Draw a sample chain length from a normal distribution"""
        size = n_samples or 1 
        x = np.random.normal(loc=mu, scale=sigma, size=size)
        chain = int(np.floor(x))
        return chain

    def init_arrays(self, chain_length):
        """Initialize carbon and chain tracking arrays"""
        self.carbon_array = np.zeros(chain_length, int)
        self.chain_array = np.zeros(chain_length + 1, int)
        self.chain_array[1:-1] = 1
        # -1 means not bound, (i,j) tuple means bound at that site
        self.carbon_to_site = np.full(chain_length, -1, dtype=object)
        # Track which carbon is at each site (inverse mapping)
        self.site_to_carbon = {}  # {(row, col): carbon_index}
        
    def init_active_sites(self, m_size):
        n = m_size
        total_sites = n * n
        
        # Two separate matrices
        self.m_bond  = np.zeros((n, n), int)  # bond type (0,1,2,3,4)
        self.m_chain = np.zeros((n, n), int)  # chain length (+N, -N, 0)
        
        occupied_sites = np.count_nonzero(self.m_bond)
        self.theta = occupied_sites / total_sites
    
    def invalidate_chains(self):
        """Mark chain cache as invalid - call when cracking happens"""
        self._chains_valid = False
