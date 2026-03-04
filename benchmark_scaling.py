import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from kmc_new.simulation import run_simulation


N_REPEATS = 3  # Repetitions per benchmark point for averaging


def benchmark_scaling(parameter_name, parameter_values, fixed_params):
    """
    Benchmarks time and RAM scaling with averaging over N_REPEATS runs.

    Args:
        parameter_name: 'm_size' or 'chain_length'
        parameter_values: list of values to sweep
        fixed_params: dict of fixed simulation parameters

    Returns:
        times (mean wall time per point), memories (mean peak RAM in MB),
        time_stds, mem_stds (standard deviations across repeats)
    """
    times, memories = [], []
    time_stds, mem_stds = [], []

    print(f"\nBenchmarking {parameter_name} scaling...")

    for val in parameter_values:
        current_params = fixed_params.copy()
        current_params[parameter_name] = val

        run_times, run_mems = [], []

        for rep in range(N_REPEATS):
            tracemalloc.start()
            t0 = time.perf_counter()

            run_simulation(**current_params)

            elapsed = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            run_times.append(elapsed)
            run_mems.append(peak_mem / 1024**2)  # bytes → MB

        mean_t = np.mean(run_times)
        mean_m = np.mean(run_mems)
        std_t  = np.std(run_times)
        std_m  = np.std(run_mems)

        times.append(mean_t)
        memories.append(mean_m)
        time_stds.append(std_t)
        mem_stds.append(std_m)

        print(f"  {parameter_name}={val:>6}: "
              f"{mean_t:.3f} ± {std_t:.3f} s  |  "
              f"{mean_m:.2f} ± {std_m:.2f} MB")

    return (np.array(times), np.array(memories),
            np.array(time_stds), np.array(mem_stds))


def fit_scaling_exponent(x, y):
    """Fit log-log slope to determine scaling exponent O(x^alpha)."""
    log_x = np.log(np.array(x, dtype=float))
    log_y = np.log(np.array(y, dtype=float))
    coeffs = np.polyfit(log_x, log_y, 1)
    return coeffs[0]  # exponent alpha


def plot_panel(ax, x, y, yerr, xlabel, ylabel, title, color,
               show_loglog=True, exponent=None):
    """Plot a single benchmark panel with error bars and optional log-log inset."""
    ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color,
                capsize=4, linewidth=2, markersize=6)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    if exponent is not None:
        ax.text(0.05, 0.92, f'scaling ∝ $x^{{{exponent:.2f}}}$',
                transform=ax.transAxes, fontsize=11,
                color=color, fontweight='bold')


def plot_loglog_panel(ax, x, y, xlabel, ylabel, title, color, exponent):
    """Log-log version of a panel to make scaling class visible."""
    ax.loglog(x, y, 'o-', color=color, linewidth=2, markersize=6)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title + ' [log-log]', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.text(0.05, 0.92, f'slope ≈ {exponent:.2f}',
            transform=ax.transAxes, fontsize=11,
            color=color, fontweight='bold')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
lattice_sizes = [5, 10, 20, 40, 60]          # m_size values (N×N surface)
alkane_sizes  = [100, 500, 1000, 5000, 10000] # chain_length values

# max_steps normalises work across system sizes — avoids apples-to-oranges
# comparison that reaction_time alone would give (larger systems do more
# steps per unit simulated time)
fixed_for_lattice = {
    'chain_length': 500,
    'reaction_time': 100,
    'temp_C': 250,
    'max_steps': 500,   # cap at fixed number of KMC steps
}
fixed_for_alkane = {
    'm_size': 10,
    'reaction_time': 100,
    'temp_C': 250,
    'max_steps': 500,
}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
lat_times, lat_mems, lat_t_std, lat_m_std = benchmark_scaling(
    'm_size', lattice_sizes, fixed_for_lattice
)
alk_times, alk_mems, alk_t_std, alk_m_std = benchmark_scaling(
    'chain_length', alkane_sizes, fixed_for_alkane
)

# Relative memory (subtract baseline to isolate growth)
lat_mems_rel = lat_mems - lat_mems[0]
alk_mems_rel = alk_mems - alk_mems[0]

# x-axis: total lattice sites N²
lat_sites = np.array(lattice_sizes) ** 2

# Scaling exponents
exp_lat_time = fit_scaling_exponent(lat_sites, lat_times)
exp_lat_mem  = fit_scaling_exponent(lat_sites, lat_mems.clip(min=1e-6))
exp_alk_time = fit_scaling_exponent(alkane_sizes, alk_times)
exp_alk_mem  = fit_scaling_exponent(alkane_sizes, alk_mems.clip(min=1e-6))

print(f"\nScaling exponents:")
print(f"  Lattice  — time: O(N^{exp_lat_time:.2f})  |  RAM: O(N^{exp_lat_mem:.2f})")
print(f"  Alkane   — time: O(L^{exp_alk_time:.2f})  |  RAM: O(L^{exp_alk_mem:.2f})")

# ---------------------------------------------------------------------------
# Plotting  (2 rows × 4 columns: linear + log-log for each metric)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('KMC Scaling Benchmark', fontsize=16, fontweight='bold', y=1.01)

# --- Row 0: Lattice scaling ---
plot_panel(axs[0, 0], lat_sites, lat_times, lat_t_std,
           'Total Sites ($N^2$)', 'Mean Time (s)',
           'Time vs Lattice Sites', 'steelblue', exponent=exp_lat_time)

plot_loglog_panel(axs[0, 1], lat_sites, lat_times,
                  'Total Sites ($N^2$)', 'Mean Time (s)',
                  'Time vs Lattice Sites', 'steelblue', exp_lat_time)

plot_panel(axs[0, 2], lat_sites, lat_mems_rel, lat_m_std,
           'Total Sites ($N^2$)', 'Relative Peak RAM (MB)',
           'RAM growth vs Lattice Sites', 'tomato', exponent=exp_lat_mem)

plot_loglog_panel(axs[0, 3], lat_sites, lat_mems.clip(min=1e-6),
                  'Total Sites ($N^2$)', 'Peak RAM (MB)',
                  'RAM vs Lattice Sites', 'tomato', exp_lat_mem)

# --- Row 1: Alkane scaling ---
plot_panel(axs[1, 0], alkane_sizes, alk_times, alk_t_std,
           'Chain Length ($L$)', 'Mean Time (s)',
           'Time vs Alkane Length', 'seagreen', exponent=exp_alk_time)

plot_loglog_panel(axs[1, 1], alkane_sizes, alk_times,
                  'Chain Length ($L$)', 'Mean Time (s)',
                  'Time vs Alkane Length', 'seagreen', exp_alk_time)

plot_panel(axs[1, 2], alkane_sizes, alk_mems_rel, alk_m_std,
           'Chain Length ($L$)', 'Relative Peak RAM (MB)',
           'RAM growth vs Alkane Length', 'mediumpurple', exponent=exp_alk_mem)

plot_loglog_panel(axs[1, 3], alkane_sizes, alk_mems.clip(min=1e-6),
                  'Chain Length ($L$)', 'Peak RAM (MB)',
                  'RAM vs Alkane Length', 'mediumpurple', exp_alk_mem)

plt.tight_layout()
plt.savefig('kmc_scaling_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nPlot saved to kmc_scaling_performance.png")