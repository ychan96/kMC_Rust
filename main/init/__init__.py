# __init__.py
from .init import BaseKineticMC
from .config import ConfigMixin
from .reactions import ReactionMixin
from .simulation import run_simulation, run_multiple_simulations
from .utils import plot_comparison, identify_final_products

# Complete KMC class
class KineticMC(BaseKineticMC, ConfigMixin, ReactionMixin):
    """
    Kinetic Monte Carlo simulation for hydrocarbon chain reactions.
    
    This class combines the base functionality with configuration and reaction 
    capabilities to simulate the kinetics of carbon chain formation and cracking.
    """

    pass

# Export 
__all__ = ['KineticMC', 'run_simulation', 'run_multiple_simulations', 
           'identify_final_products', 'plot_product_distribution']