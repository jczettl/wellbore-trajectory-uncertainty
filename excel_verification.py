"""Validate the implementation against the three ISCWSA example workbooks.

The trajectory is always reconstructed with Minimum Curvature. Covariance
sensitivities can use either the benchmark Balanced Tangential (BT) equations
or the analytical Minimum Curvature (MCM) equations.
"""

import numpy as np

from excel_extraction import run_extraction
from excel_verification_metrics import (
    compare_covariances,
    covariance_max_absolute_error,
    covariance_rms_relative_error,
    validate_tvd,
)
from excel_verification_visuals import plot_tvd_validation
from iscwsa_error import calculate_iscwsa_covariance
from trajectory import calculate_spatial_trajectory


BENCHMARK_FILES = tuple(
    f"data/error-model-example-mwdrev5-1-iscwsa-{number}.xlsx"
    for number in (1, 2, 3)
)

ERROR_MODEL = {
    "DRFR": {"mag": 0.35, "axis": "MD", "mode": "R",
             "weight": lambda row, prev, dl: 1.0},
    "DSFS": {"mag": 0.00056, "axis": "MD", "mode": "S",
             "weight": lambda row, prev, dl: dl},
    "DSTG": {"mag": 0.00000025, "axis": "MD", "mode": "G",
             "weight": lambda row, prev, dl:
                 row["MD"] * row["TVD"] - prev["MD"] * prev["TVD"]},
    "DEC-U": {"mag": 0.16, "axis": "Azi", "mode": "W",
              "weight": lambda row, prev, dl: 1.0},
    "DEC-OS": {"mag": 0.24, "axis": "Azi", "mode": "G",
               "weight": lambda row, prev, dl: 1.0},
    "DEC-OH": {"mag": 0.21, "axis": "Azi", "mode": "G",
               "weight": lambda row, prev, dl: 1.0},
    "DEC-OI": {"mag": 0.05, "axis": "Azi", "mode": "G",
               "weight": lambda row, prev, dl: 1.0},
    "DECR": {"mag": 0.1, "axis": "Azi", "mode": "R",
             "weight": lambda row, prev, dl: 1.0},
    "SAGE": {"mag": 0.2, "axis": "Inc", "mode": "S",
             "weight": lambda row, prev, dl:
                 max(0.0, np.sin(np.radians(row["Incl"]))) ** 0.25},
    "XYM1": {"mag": 0.1, "axis": "Inc", "mode": "S",
             "weight": lambda row, prev, dl:
                 abs(np.sin(np.radians(row["Incl"])))},
    "XYM2": {"mag": 0.1, "axis": "Azi", "mode": "S",
             "weight": lambda row, prev, dl: -1.0},
}


def run_pipeline(file_path, balanced_tangential_jacobi_approximation=False):
    """Return reference and calculated results for one benchmark workbook."""
    extracted = run_extraction(file_path, ERROR_MODEL)
    if extracted is None:
        return None

    wellpath, error_model, reference_covariances = extracted
    if wellpath is None:
        return None

    trajectory = calculate_spatial_trajectory(
        wellpath,
        md_col="MD",
        inc_col="Incl",
        azi_col="Azi",
        ns_col="NS",
        ew_col="EW",
        tvd_col="TVD",
        start_coords=(0.0, 0.0, 0.0),
    )
    _, term_covariances, _ = calculate_iscwsa_covariance(
        trajectory,
        error_model,
        md_col="MD",
        inc_col="Incl",
        azi_col="Azi",
        balanced_tangential_jacobi_approximation=(
            balanced_tangential_jacobi_approximation
        ),
    )
    return wellpath, trajectory, term_covariances, reference_covariances


def _print_ignored_formula_rows(reference_covariances):
    """Report workbook formula rows excluded during MD alignment."""
    counts = {}
    for reference in reference_covariances.values():
        rows = int(reference.attrs.get("ignored_trailing_formula_rows", 0))
        if rows:
            counts[rows] = counts.get(rows, 0) + 1
    if counts:
        details = ", ".join(
            f"{rows} row(s) in {sheets} sheet(s)"
            for rows, sheets in sorted(counts.items())
        )
        print(f"Excluded trailing formula rows after MD alignment: {details}.")


def verify_workbook(file_path, use_bt_sensitivities=True, plot_tvd=False):
    """Print a compact TVD and covariance comparison for one workbook."""
    data = run_pipeline(
        file_path,
        balanced_tangential_jacobi_approximation=use_bt_sensitivities,
    )
    if data is None:
        raise RuntimeError(f"Benchmark extraction failed: {file_path}")

    reference, calculated, term_covariances, reference_covariances = data
    # Run structural, finite-value, and MD-alignment checks before summarizing.
    validate_tvd(calculated, reference, tolerance=np.inf)
    compare_covariances(
        calculated,
        term_covariances,
        reference_covariances,
        tolerance=np.inf,
    )
    residual = calculated["TVD"].to_numpy() - reference["TVD"].to_numpy()
    maximum_index = int(np.argmax(np.abs(residual)))

    rms = covariance_rms_relative_error(
        term_covariances, reference_covariances,
    )[["Error Term", "Overall RMS"]]
    maximum = covariance_max_absolute_error(
        term_covariances, reference_covariances,
    )[["Error Term", "Overall Max Diff"]]
    covariance_summary = (
        maximum.merge(rms, on="Error Term")
        .sort_values("Overall Max Diff", ascending=False)
    )

    method = "BT" if use_bt_sensitivities else "MCM"
    print(f"\nISCWSA BENCHMARK: {file_path}")
    print("-" * (18 + len(file_path)))
    print(f"Covariance sensitivities: {method}")
    print(f"Stations:                {len(reference)}")
    _print_ignored_formula_rows(reference_covariances)
    print("\nTVD RESIDUALS (calculated - reference)")
    print(f"Maximum absolute: {abs(residual[maximum_index]):.6e} m")
    print(f"MD at maximum:   {calculated['MD'].iloc[maximum_index]:.2f} m")
    print(f"Signed EOW:      {residual[-1]:.6e} m")
    print("\nCOVARIANCE RESIDUALS")
    print(covariance_summary.to_string(index=False, float_format="{:.6e}".format))

    if plot_tvd:
        plot_tvd_validation(calculated, remove_baseline=False, save_filename=None)

    return {
        "tvd_residual": residual,
        "covariance_summary": covariance_summary,
    }


def run_verification(file_paths=BENCHMARK_FILES, use_bt_sensitivities=True,
                     plot_tvd=False):
    """Run the same verification for each selected workbook."""
    return {
        file_path: verify_workbook(file_path, use_bt_sensitivities, plot_tvd)
        for file_path in file_paths
    }


if __name__ == "__main__":
    run_verification(
        file_paths=BENCHMARK_FILES,
        use_bt_sensitivities=True,
        plot_tvd=False,
    )
