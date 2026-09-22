"""Optional reproduction of report Figure 3's two finite-difference plots."""

from pathlib import Path

import matplotlib.pyplot as plt

from jacobian_validation import run_validation


def save_plot(fig, ax, output_dir, filename, title, xlabel):
    """Apply shared labels and save one validation figure."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maximum absolute Jacobian-entry difference")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {path}")


def create_plots(jacobian, convergence, output_dir="jacobian_validation_results"):
    """Save the stationwise FD errors and representative-leg convergence curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in ("mcm", "bt"):
        ax.plot(jacobian["MD_end"], jacobian[f"{method}_max_abs_error"],
                label=f"{method.upper()} FD vs analytical")
    ax.set_yscale("log")
    ax.legend()
    save_plot(fig, ax, output_dir, "01_fd_jacobian_error_vs_md.png",
              "Finite-Difference Jacobian Validation", "Measured Depth (m)")

    if not convergence.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for leg, group in convergence.groupby("leg_index"):
            ax.loglog(group["step_multiplier"], group["mcm_max_abs_error"],
                      marker="o", label=f"MCM leg {leg}")
            ax.loglog(group["step_multiplier"], group["bt_max_abs_error"],
                      marker="x", linestyle="--", label=f"BT leg {leg}")
        ax.legend(fontsize=8)
        save_plot(fig, ax, output_dir, "06_fd_convergence.png",
                  "Finite-Difference Step-Size Convergence", "Finite-difference step multiplier")


if __name__ == "__main__":
    jacobian, convergence = run_validation(
        csv_path="data/15_9_F_11_A.csv",
        fd_steps={"inc": 1e-3, "azi": 1e-3, "dl": 0.1},
        step_multipliers=(10, 1, 0.1, 0.01, 0.001),
    )
    create_plots(jacobian, convergence, output_dir="jacobian_validation_plot")
