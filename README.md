# kMC Hydrocarbon Cracking Simulator

A high-performance kinetic Monte Carlo (kMC) simulation framework for modeling hydrocarbon cracking and hydrogenolysis of long-chain alkanes (C300+) on metal catalyst surfaces.

## Overview

This project combines Rust's performance with Python's scientific ecosystem to simulate realistic catalytic reaction kinetics on metal surfaces like Pt(111) and Pt(100).

**Key Features:**
- **Hybrid Rust/Python architecture**: Rust core for performance-critical kMC loops, Python for analysis and visualization
- **Physical catalyst geometry**: Explicit Cartesian coordinates with atop/hollow/bridge site types
- **Chain-length dependent kinetics**: Realistic van der Waals scaling and position-dependent reaction rates
- **Flexible hydrogen model**: Langmuir isotherm-based H coverage with optional explicit blocking
- **Product distribution fitting**: Compare simulations against experimental hydrocarbon product data

## Architecture
```
kmc_rust/
├── src/                    # Rust core (performance-critical)
│   ├── lib.rs             # PyO3 bindings
│   ├── init.rs            # KineticMC struct, parameters
│   ├── catalyst.rs        # Surface geometry, neighbor maps
│   ├── reactions.rs       # Reaction execution
│   └── site_counter.rs    # Available site counting
│
└── python/                 # Python wrapper (analysis/viz)
    ├── main.py            # CLI entry point
    ├── simulation.py      # Simulation orchestration
    ├── plotting.py        # Matplotlib visualizations
    └── analysis.py        # Product statistics
```

## Physics Model

### Reaction Network
- **Adsorption/Desorption**: Gas-phase ↔ surface equilibrium
- **Double M-C formation**: Dehydrogenation to form dMC anchors
- **C-C Scission**: Bond cleavage after dMC formation
- Chain-length categories: C1, C2, C3, C4, C5+ (internal/terminal)

### Rate Expressions
- **Desorption**: `k_d(N) = A_d · exp(-(E₀ + α_vdw·N) / k_B T)` (van der Waals scaling)
- **Adsorption**: `k_ads · (1 - θ_H)` (H-blocked site correction)
- **dMC/Scission**: Arrhenius with internal bond penalty `β_int`

### Catalyst Surface
- **Geometry**: FCC(111) hexagonal or FCC(100) square lattice
- **Site types**: Atop (C adsorption), Hollow (H blocking), Bridge (inactive)
- **Neighbor maps**: Pre-computed for efficient dMC partner search

## Installation

### Prerequisites
- **Rust** (1.70+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Python** (3.9+): with `pip`
- **maturin**: `pip install maturin`

### Build & Install
```bash
git clone https://github.com/yourusername/kmc_rust.git
cd kmc_rust

# Build Rust extension
maturin develop --release

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Single Simulation
```python
import kmc_rust

kmc = kmc_rust.PyKineticMC(
    temp_c=250.0,
    reaction_time=7200.0,
    chain_length=None,  # Random from N(280, 10)
    p_h2=40.0,
    m_size=20
)

while kmc.get_time() < 7200.0:
    kmc.step()

products = kmc.get_products()
print(f"Product distribution: {products}")
```

### Command Line
```bash
python main.py --temp 250 --time 7200 --sims 100 --msize 20
```

**Options:**
- `--temp`: Temperature (°C)
- `--time`: Reaction time (seconds)
- `--sims`: Number of simulations
- `--length`: Initial chain length (default: random N(280, 10))
- `--msize`: Catalyst surface grid size
- `--coverage`: Track surface coverage and generate GIF
- `--exp-data`: Path to experimental data for comparison

### Parameter Optimization
```python
from kmc.optimize import run_bayesian_optimization

best_params = run_bayesian_optimization(
    experimental_data="data.xlsx",
    n_trials=100
)
```

## Configuration

### Catalyst Geometry (JSON)
```json
{
  "geometry": {
    "metal": "Pt",
    "facet": "111",
    "lattice_constant": 3.92,
    "dimensions": [20, 20],
    "periodic": [true, true]
  },
  "c_site_type": "atop",
  "h_site_type": "hollow",
  "neighbor_cutoff": 4.0
}
```

### Kinetic Parameters
9 core parameters control all reaction rates:
```rust
k_ads       // Adsorption rate constant
A_des       // Desorption pre-exponential
E0_des      // Base desorption energy (C1)
alpha_vdw   // VdW scaling per carbon
A_scission  // Scission pre-exponential
E_dMC       // dMC formation barrier
E_scission  // Scission barrier
beta_int    // Internal bond penalty
K_H2        // H2 equilibrium constant
```

## Output

### Product Distribution Plot
Compares simulation vs. experimental mass-based distributions:
![Product Distribution](examples/product_distribution.png)

### Surface Coverage Animation
GIF showing evolution of adsorbed species:
![Coverage Animation](examples/coverage_animation.gif)

## Development

### Run Tests
```bash
# Rust tests
cargo test

# Python tests
pytest tests/
```

### Code Structure
- **init.rs**: Core kMC data structures
- **catalyst.rs**: Physical surface geometry
- **reactions.rs**: Reaction logic (adsorption, desorption, cracking)
- **site_counter.rs**: Available site enumeration
- **lib.rs**: Python bindings via PyO3

## Performance

**Benchmarks** (single simulation, C280 → products, 7200s):
- Python-only: ~120 seconds
- Rust core: ~5 seconds (**24x speedup**)

Tested on: Apple M1, 16GB RAM

## Citation

If you use this code in research, please cite:
```
[Your paper citation here]
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contact

- **Author**: ychan96
- **Email**: ethanparkerbros@gmail.com
- **Issues**: https://github.com/ychan96/kmc_rust/issues

## Acknowledgments

- Physical chemistry guidance: kyeongsu@kist.re.kr
- Computational methods: Based on Gillespie algorithm for stochastic simulation
