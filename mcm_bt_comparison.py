"""Compare MCM and BT sensitivities and print the report's three result tables.

Run: python mcm_bt_comparison.py
Change the survey and settings at the bottom of this file. No files are saved.
"""

import numpy as np
import pandas as pd

from core_math import get_ellipse_parameters
from iscwsa_error import calculate_iscwsa_covariance
from jacobian_validation import STANDARD_ERROR_MODEL
from reporting import generate_jacobian_comparison_table
from survey_qc import validate_and_clean_survey
from trajectory import calculate_spatial_trajectory


def load_physical_survey(csv_path):
    """Read the CSV and keep validated, physical survey stations."""
    survey = validate_and_clean_survey(pd.read_csv(csv_path))
    if "is_interpolated" in survey.columns:
        survey = survey.loc[~survey["is_interpolated"].astype(bool)]
    survey = survey.reset_index(drop=True)
    if len(survey) < 2:
        raise ValueError("At least two physical survey stations are required.")
    return survey


def calculate_covariance_pair(survey, error_model=None):
    """Use one MCM trajectory and error model with the two sensitivity methods."""
    if error_model is None:
        error_model = STANDARD_ERROR_MODEL
    start_coords = tuple(
        float(survey[name].iloc[0]) if name in survey.columns else 0.0
        for name in ("NS", "EW", "TVD")
    )
    trajectory = calculate_spatial_trajectory(survey, start_coords=start_coords)
    # The shared DSTG formula expects this name for the reconstructed TVD.
    trajectory["TVD_calc_validation"] = trajectory["TVD"]

    covariance_mcm, _, _ = calculate_iscwsa_covariance(
        trajectory, error_model, balanced_tangential_jacobi_approximation=False,
    )
    covariance_bt, _, _ = calculate_iscwsa_covariance(
        trajectory, error_model, balanced_tangential_jacobi_approximation=True,
    )
    for name, covariance in (("MCM", covariance_mcm), ("BT", covariance_bt)):
        if covariance.shape != (len(trajectory), 3, 3) or not np.isfinite(covariance).all():
            raise ValueError(f"{name} covariance has an invalid shape or non-finite values.")
        transpose = covariance.swapaxes(1, 2)
        scale = max(1.0, float(np.max(np.abs(covariance))))
        if np.max(np.abs(covariance - transpose)) > 1e-12 * scale:
            raise ValueError(f"{name} covariance is not symmetric.")
        if np.min(np.linalg.eigvalsh((covariance + transpose) / 2)) < -1e-10 * scale:
            raise ValueError(f"{name} covariance is not positive semidefinite.")
    return trajectory, covariance_mcm, covariance_bt


def relative_rms(difference, reference):
    """Use the complete BT profile as one reference scale (a ratio, not percent)."""
    numerator = np.linalg.norm(np.asarray(difference).ravel())
    denominator = np.linalg.norm(np.asarray(reference).ravel())
    epsilon = np.finfo(float).eps
    if denominator > epsilon:
        return numerator / denominator
    return 0.0 if numerator <= epsilon else np.nan


def profile_statistics(mcm, bt, md, difference=None, valid=None, relative=True):
    """Summarize a scalar profile; retain the sign of the end-of-well difference."""
    if difference is None:
        difference = mcm - bt
    if valid is None:
        valid = np.ones(len(md), dtype=bool)
    valid = valid & np.isfinite(mcm) & np.isfinite(bt) & np.isfinite(difference)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return dict.fromkeys([
            "rms_difference", "maximum_abs_difference", "global_relative_rms",
            "eow_difference", "md_at_maximum_m",
        ], np.nan)
    selected = difference[valid]
    maximum_index = indices[np.argmax(np.abs(selected))]
    return {
        "rms_difference": np.sqrt(np.mean(selected ** 2)),
        "maximum_abs_difference": np.max(np.abs(selected)),
        "global_relative_rms": relative_rms(selected, bt[valid]) if relative else np.nan,
        "eow_difference": difference[-1] if valid[-1] else np.nan,
        "md_at_maximum_m": md[maximum_index],
    }


def compare_covariance_elements(trajectory, covariance_mcm, covariance_bt):
    """Compare the six independent covariance elements (report Table 10)."""
    md = trajectory["MD"].to_numpy()
    components = {"NN": (0, 0), "EE": (1, 1), "VV": (2, 2),
                  "NE": (0, 1), "NV": (0, 2), "EV": (1, 2)}
    rows = []
    for component, (i, j) in components.items():
        stats = profile_statistics(covariance_mcm[:, i, j], covariance_bt[:, i, j], md)
        rows.append({"component": component, "unit": "m^2", **stats})
    return pd.DataFrame(rows)


def uncertainty_profiles(covariance, sigma_level):
    """Extract NE axes, orientation in radians, vertical extent, and 3D axes."""
    major, minor, angle = get_ellipse_parameters(
        covariance, plane="NE", scale_mode="sigma", sigma_level=sigma_level,
    )
    axes_3d, _ = get_ellipse_parameters(
        covariance, plane="3D", scale_mode="sigma", sigma_level=sigma_level,
    )
    return {
        "NE_semi_major": major,
        "NE_semi_minor": minor,
        "NE_major_axis_azimuth": angle,
        "vertical_marginal": sigma_level * np.sqrt(np.clip(covariance[:, 2, 2], 0, None)),
        "axis_3D_smallest": axes_3d[:, 0],
        "axis_3D_middle": axes_3d[:, 1],
        "axis_3D_largest": axes_3d[:, 2],
    }


def compare_uncertainty(trajectory, covariance_mcm, covariance_bt,
                        sigma_level=2.0, orientation_anisotropy_threshold=0.01):
    """Return report Table 7 and the difference profiles used by Figure 4."""
    if not np.isfinite(sigma_level) or sigma_level <= 0:
        raise ValueError("sigma_level must be finite and positive.")
    if not np.isfinite(orientation_anisotropy_threshold) or orientation_anisotropy_threshold < 0:
        raise ValueError("orientation_anisotropy_threshold must be finite and non-negative.")
    md = trajectory["MD"].to_numpy()
    mcm = uncertainty_profiles(covariance_mcm, sigma_level)
    bt = uncertainty_profiles(covariance_bt, sigma_level)
    differences = pd.DataFrame({"MD": md})
    rows = []
    for metric in mcm:
        if metric == "NE_major_axis_azimuth":
            # An ellipse axis is unchanged by a 180-degree rotation.
            raw_angle = mcm[metric] - bt[metric]
            difference = np.degrees(np.abs((raw_angle + np.pi / 2) % np.pi - np.pi / 2))
            valid = np.ones(len(md), dtype=bool)
            for profiles in (mcm, bt):
                major, minor = profiles["NE_semi_major"], profiles["NE_semi_minor"]
                epsilon = np.finfo(float).eps
                anisotropy = (major - minor) / np.maximum(major, epsilon)
                valid &= (major > epsilon) & (anisotropy >= orientation_anisotropy_threshold)
            stats = profile_statistics(
                np.degrees(mcm[metric]) % 180, np.degrees(bt[metric]) % 180, md,
                difference=difference, valid=valid, relative=False,
            )
            differences[metric] = np.where(valid, difference, np.nan)
            unit = "deg"
        else:
            stats = profile_statistics(mcm[metric], bt[metric], md)
            differences[metric] = mcm[metric] - bt[metric]
            unit = "m"
        rows.append({"metric": metric, "unit": unit, **stats})

    # Compare matrices with the Frobenius norm, not the difference of their norms.
    covariance_difference = covariance_mcm - covariance_bt
    frobenius_difference = np.linalg.norm(covariance_difference, axis=(1, 2))
    stats = profile_statistics(
        np.linalg.norm(covariance_mcm, axis=(1, 2)),
        np.linalg.norm(covariance_bt, axis=(1, 2)), md,
        difference=frobenius_difference,
    )
    stats["global_relative_rms"] = relative_rms(covariance_difference, covariance_bt)
    rows.append({"metric": "full_covariance", "unit": "m^2", **stats})
    differences["full_covariance"] = frobenius_difference
    return pd.DataFrame(rows), differences


def print_table(title, table, columns):
    """Print compact tables with the same quantities and units as the report."""
    labels = {
        "parameter": "Column", "component": "Element", "metric": "Output", "unit": "Unit",
        "rms_difference": "RMS", "percentile_95_difference": "95th percentile",
        "maximum_difference": "Maximum", "maximum_abs_difference": "Maximum",
        "relative_rms": "Relative RMS", "global_relative_rms": "Relative RMS",
        "eow_difference": "EOW difference", "md_at_maximum_m": "MD at max (m)",
    }
    display = table[columns].rename(columns=labels)
    print("\n" + title)
    print("-" * len(title))
    print(display.to_string(index=False, float_format="{:.6e}".format,
                            formatters={"MD at max (m)": "{:.2f}".format}, na_rep="--"))


def run_comparison(csv_path, sigma_level=2.0, orientation_anisotropy_threshold=0.01):
    """Calculate and print all comparison results without writing output files."""
    survey = load_physical_survey(csv_path)
    print(f"\nSurvey: {csv_path} | Physical stations: {len(survey)}")
    trajectory, covariance_mcm, covariance_bt = calculate_covariance_pair(survey)
    jacobian = generate_jacobian_comparison_table(trajectory, output_dir=None, print_table=False)
    covariance = compare_covariance_elements(trajectory, covariance_mcm, covariance_bt)
    uncertainty, differences = compare_uncertainty(
        trajectory, covariance_mcm, covariance_bt, sigma_level, orientation_anisotropy_threshold,
    )
    print("Relative RMS is a fraction using the complete BT profile as reference.")
    print("Differences are MCM - BT; orientation and full covariance use magnitudes.")
    print(f"Uncertainty axes use {sigma_level:g} sigma; NE orientation needs anisotropy >= "
          f"{orientation_anisotropy_threshold:g} in both ellipses.")
    print_table("JACOBIAN COLUMNS (Table 6)", jacobian, [
        "parameter", "unit", "rms_difference", "percentile_95_difference",
        "maximum_difference", "relative_rms", "md_at_maximum_m",
    ])
    summary_columns = ["unit", "rms_difference", "maximum_abs_difference",
                       "global_relative_rms", "eow_difference", "md_at_maximum_m"]
    print_table("COVARIANCE ELEMENTS (Table 10)", covariance, ["component"] + summary_columns)
    print_table("DERIVED UNCERTAINTY (Table 7)", uncertainty, ["metric"] + summary_columns)
    return {"jacobian": jacobian, "covariance": covariance,
            "uncertainty": uncertainty, "differences": differences}


if __name__ == "__main__":
    run_comparison(
        csv_path="data/15_9_F_11_A.csv",
        sigma_level=2.0,
        orientation_anisotropy_threshold=0.01,
    )
