"""Optional reproduction of report Figure 4; the comparison script stays console-only."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from mcm_bt_comparison import (
    load_physical_survey, calculate_covariance_pair, compare_uncertainty,
)


def plot_comparison(csv_path, output_path="propagated_uncertainty_difference.png",
                    sigma_level=2.0, orientation_anisotropy_threshold=0.01):
    """Save the axis-length, covariance, and axial-angle difference panels."""
    survey = load_physical_survey(csv_path)
    trajectory, covariance_mcm, covariance_bt = calculate_covariance_pair(survey)
    _, differences = compare_uncertainty(
        trajectory, covariance_mcm, covariance_bt, sigma_level, orientation_anisotropy_threshold,
    )
    md = differences["MD"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11), sharex=True)
    for metric, label in [("NE_semi_major", "NE semi-major"),
                          ("NE_semi_minor", "NE semi-minor"),
                          ("vertical_marginal", "Vertical marginal"),
                          ("axis_3D_largest", "Largest 3D axis")]:
        axes[0].plot(md, np.abs(differences[metric]), label=label)
    axes[0].set_ylabel("Absolute difference (m)")
    axes[0].set_title(f"MCM versus BT propagated {sigma_level:.3g}σ uncertainty differences")
    axes[0].legend(ncol=2)
    axes[1].plot(md, differences["full_covariance"])
    axes[1].set_ylabel(r"$\|C_{\mathrm{MCM}}-C_{\mathrm{BT}}\|_F$ (m$^2$)")
    valid = differences["NE_major_axis_azimuth"].notna()
    axes[2].plot(md[valid], differences.loc[valid, "NE_major_axis_azimuth"],
                 marker=".", markersize=3, linewidth=1)
    axes[2].set_xlabel("Measured Depth (m)")
    axes[2].set_ylabel("NE axial-angle difference (deg)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    plot_comparison(
        csv_path="data/15_9_F_11_A.csv",
        output_path="mcm_bt_comparison_plot/propagated_uncertainty_difference.png",
        sigma_level=2.0,
        orientation_anisotropy_threshold=0.01,
    )
