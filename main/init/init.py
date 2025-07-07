import numpy as np
from scipy import constants

eV = 1.60218e-19

class BaseKineticMC:
    def __init__(self, temp_C=250, reaction_time=7200, chain_length=20, rate_constants=None,P_H2=50):
        #Parameters
        self.temp_K = temp_C + 273.15
        self.kb_eV  = constants.Boltzmann / constants.e
        self.reaction_time = reaction_time
        #Initialize arrays
        self.init_arrays(chain_length)
        # Ea, pre_A, k_const
        #self.Ea = np.array([0.31, 0.31, 1.04, 1.04, 0.18, 0.18, 10.15, 0.89, 0.89]) * eV

        #if pre_A is None:
        #    self.pre_A = np.array([10, 0.1, 10, 10, 10, 10, 0.00001, 0.003, 0.001])
        #else:
        #    self.pre_A = np.array(pre_A)
        if rate_constants is not None:
            self.k_const = np.array(rate_constants)
        else:
            self.k_const = np.array([
                1.0e-3,  # Internal adsorption
                1.0e-4,  # Terminal adsorption
                1.0e-5,  # Internal desorption
                1.0e-5,  # Terminal desorption
                1.0e-3,  # Internal dehydrogenation
                1.0e-3,  # Terminal dehydrogenation
                1.0e-8,  # Double M-C desorption
                1.0e-6,  # Internal cracking
                1.0e-7   # Terminal cracking
            ])
        self.current_time = 0.0
        self._chains = None
        self._chains_valid = False

    @property #this decorator provides lazy evaluation, caching, encapsulation,  (attributes)
    def chains(self):
        if not self._chains_valid or self._chains is None:
            self._chains = self._identify_chains()
            self._chains_valid = True
        return self._chains


    def init_arrays(self, chain_length):
        self.carbon_array = np.zeros(chain_length, int)
        self.chain_array  = np.zeros(chain_length+1, int)
        self.chain_array[1:-1] = 1

    #def rate_constants(self):
        #return self.pre_A * np.exp(-self.Ea/(self.kb_eV*self.temp_K))
    
    def invalidate_chains(self):
        self._chains_valid = False