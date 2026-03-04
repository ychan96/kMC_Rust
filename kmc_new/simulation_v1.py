import numpy as np
from kmc_new.init import BaseKineticMC
from kmc_new.count_sites import ConfigMixin
from kmc_new.reactions import ReactionMixin
from kmc_new.coverage import CoverageMixin

class KMC(BaseKineticMC, ConfigMixin, ReactionMixin, CoverageMixin):
    pass

# Initialize
kmc = KMC(temp_C=250, chain_length=300, m_size=5)

# Main loop
while kmc.current_time < kmc.reaction_time:
    # Count available sites
    counts = kmc.update_configuration()
    
    # Select reaction
    reaction_key, dt = kmc.select_reaction(counts)
    
    if reaction_key is None:
        break
    
    # Perform reaction
    success, chain_info = kmc.perform_reaction(reaction_key)
    
    if success:
        # Update surface
        kmc.metal_surface(reaction_key, chain_info)
        # Update time
        kmc.current_time += dt

print(f"Final time: {kmc.current_time}")
print(f"Surface coverage: {kmc.theta}")