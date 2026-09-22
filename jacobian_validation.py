"""Check analytical MCM and BT Jacobians against central finite differences.

Run: python jacobian_validation.py
Settings are at the bottom. Results go to the console; no files are written.
Use plot_jacobian_validation.py for the two plots in report Figure 3.
"""

import numpy as np
import pandas as pd

from core_math import _get_tangent_vector, get_jacobian, min_curvature_step
from survey_qc import validate_and_clean_survey


DEFAULT_FD_STEPS = {"inc": 1e-3, "azi": 1e-3, "dl": 0.1}  # degrees, degrees, metres
PARAM_NAMES = ("inc1", "azi1", "inc2", "azi2", "dl")
PARAM_TO_STEP_KEY = ("inc", "azi", "inc", "azi", "dl")
RELATIVE_DENOM_EPS = 1e-12

# Retained here because mcm_bt_comparison.py imports this report configuration.
# These terms are used for covariance propagation, not for the FD checks below.
STANDARD_ERROR_MODEL = {
    "DRFR": {"mag": 0.35, "axis": "MD", "mode": "R",
             "weight": lambda row, prev, dl: 1.0},
    "DSFS": {"mag": 0.00056, "axis": "MD", "mode": "S",
             "weight": lambda row, prev, dl: dl},
    "DSTG": {"mag": 0.00000025, "axis": "MD", "mode": "G",
             "weight": lambda row, prev, dl: (
                 row["MD"] * row["TVD_calc_validation"]
                 - prev["MD"] * prev["TVD_calc_validation"])},
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
             "weight": lambda row, prev, dl: max(0.0, np.sin(np.radians(row["Incl"]))) ** 0.25},
    "XYM1": {"mag": 0.1, "axis": "Inc", "mode": "S",
             "weight": lambda row, prev, dl: np.abs(np.sin(np.radians(row["Incl"])))},
    "XYM2": {"mag": 0.1, "axis": "Azi", "mode": "S",
             "weight": lambda row, prev, dl: -1.0},
}


def balanced_tangential_step(inc1, azi1, inc2, azi2, dl):
    """BT displacement in [North, East, TVD]; input angles are degrees."""
    i1, i2 = np.radians(inc1), np.radians(inc2)
    a1, a2 = np.radians(azi1), np.radians(azi2)
    return 0.5 * dl * (_get_tangent_vector(i1, a1) + _get_tangent_vector(i2, a2))


def minimum_curvature_step_wrapper(inc1, azi1, inc2, azi2, dl):
    """Map Jacobian column order to the displacement function's argument order."""
    return min_curvature_step(inc1, inc2, azi1, azi2, dl)


def analytical_jacobian_from_params(params, *, balanced_tangential):
    """Map the same parameter order to the existing analytical Jacobian."""
    inc1, azi1, inc2, azi2, dl = map(float, params)
    return get_jacobian(
        inc1, inc2, azi1, azi2, dl,
        balanced_tangential_jacobi_approximation=balanced_tangential,
    )


def central_fd_jacobian(displacement_function, params, steps):
    """Perturb each parameter: J[:, j] = (r(p+h) - r(p-h)) / (2h).

    Angular columns are m/degree and the length column is m/m, matching get_jacobian.
    """
    params = np.asarray(list(params), dtype=float)
    if params.shape != (5,):
        raise ValueError("Expected parameters (inc1, azi1, inc2, azi2, dl).")
    jacobian = np.zeros((3, 5))
    for j, key in enumerate(PARAM_TO_STEP_KEY):
        h = float(steps[key])
        if not np.isfinite(h) or h <= 0:
            raise ValueError(f"Finite-difference step '{key}' must be finite and positive.")
        plus, minus = params.copy(), params.copy()
        plus[j] += h
        minus[j] -= h
        r_plus = np.asarray(displacement_function(*plus), dtype=float)
        r_minus = np.asarray(displacement_function(*minus), dtype=float)
        if r_plus.shape != (3,) or r_minus.shape != (3,):
            raise ValueError("Displacement must contain three coordinates.")
        jacobian[:, j] = (r_plus - r_minus) / (2.0 * h)
    return jacobian


def matrix_error_metrics(reference, estimate):
    """Compare entries, excluding near-zero references from relative metrics."""
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("Reference and estimate shapes differ.")
    difference = estimate - reference
    absolute = np.abs(difference)
    relative = absolute / np.maximum(np.abs(reference), RELATIVE_DENOM_EPS)
    meaningful = np.abs(reference) > RELATIVE_DENOM_EPS
    selected = relative[meaningful]
    return {
        "max_abs": float(np.max(absolute)),
        "rms_abs": float(np.sqrt(np.mean(difference ** 2))),
        "max_rel": float(np.max(selected)) if selected.size else 0.0,
        "rms_rel": float(np.sqrt(np.mean(selected ** 2))) if selected.size else 0.0,
        "max_abs_index": tuple(np.unravel_index(np.argmax(absolute), absolute.shape)),
    }


def validate_segment(params, fd_steps):
    """Check each analytical method against its own displacement equation."""
    results = {}
    for method, displacement in (("mcm", minimum_curvature_step_wrapper),
                                 ("bt", balanced_tangential_step)):
        analytical = analytical_jacobian_from_params(params, balanced_tangential=(method == "bt"))
        finite_difference = central_fd_jacobian(displacement, params, fd_steps)
        results[method] = matrix_error_metrics(finite_difference, analytical)
    return results


def load_well(csv_path):
    """Apply survey QC and retain only physical survey stations."""
    survey = validate_and_clean_survey(pd.read_csv(csv_path))
    if "is_interpolated" in survey.columns:
        survey = survey.loc[~survey["is_interpolated"].astype(bool)]
    survey = survey.reset_index(drop=True)
    if len(survey) < 2:
        raise ValueError("At least two physical survey stations are required.")
    return survey


def segment_parameters(survey, leg):
    """Leg 1 connects stations 0 and 1; return parameters in Jacobian order."""
    previous, current = survey.iloc[leg - 1], survey.iloc[leg]
    return (float(previous["Incl"]), float(previous["Azi"]),
            float(current["Incl"]), float(current["Azi"]),
            float(current["MD"] - previous["MD"]))


def validate_jacobians_over_well(survey, fd_steps):
    """Collect FD errors for every physical leg and locate each method's worst entry."""
    rows, worst = [], {"mcm": {"max_abs": -np.inf}, "bt": {"max_abs": -np.inf}}
    for leg in range(1, len(survey)):
        row = {"leg_index": leg, "MD_start": float(survey["MD"].iloc[leg - 1]),
               "MD_end": float(survey["MD"].iloc[leg])}
        for method, metrics in validate_segment(segment_parameters(survey, leg), fd_steps).items():
            for name in ("max_abs", "rms_abs", "max_rel", "rms_rel"):
                row[f"{method}_{name}_error"] = metrics[name]
            if metrics["max_abs"] > worst[method]["max_abs"]:
                index = metrics["max_abs_index"]
                worst[method] = {
                    "max_abs": metrics["max_abs"], "MD_start": row["MD_start"],
                    "MD_end": row["MD_end"], "matrix_index": index,
                    "jacobian_row": index[0], "jacobian_col": index[1],
                }
        rows.append(row)
    return pd.DataFrame(rows), worst


def run_fd_convergence_tests(survey, step_multipliers, representative_leg_indices, base_steps=None):
    """Repeat the checks while scaling all perturbations by the same multiplier."""
    if base_steps is None:
        base_steps = DEFAULT_FD_STEPS
    rows = []
    for leg in representative_leg_indices:
        if leg < 1 or leg >= len(survey):
            continue
        params = segment_parameters(survey, leg)
        for multiplier in step_multipliers:
            steps = {key: value * multiplier for key, value in base_steps.items()}
            row = {"leg_index": leg, "MD_start": float(survey["MD"].iloc[leg - 1]),
                   "MD_end": float(survey["MD"].iloc[leg]), "step_multiplier": multiplier,
                   "inc_step_deg": steps["inc"], "azi_step_deg": steps["azi"],
                   "dl_step_m": steps["dl"]}
            for method, metrics in validate_segment(params, steps).items():
                row[f"{method}_max_abs_error"] = metrics["max_abs"]
                row[f"{method}_rms_abs_error"] = metrics["rms_abs"]
            rows.append(row)
    return pd.DataFrame(rows)


def print_summary(jacobian, convergence, worst):
    """Print overall errors, their locations, and the convergence study."""
    print("\nANALYTICAL JACOBIANS VS CENTRAL FINITE DIFFERENCES")
    print("Angular columns: m/degree; length column: m/m.")
    print(f"Relative errors exclude FD entries with magnitude <= {RELATIVE_DENOM_EPS:g}.")
    for method in ("mcm", "bt"):
        error = worst[method]
        coordinate = ("North", "East", "TVD")[error["jacobian_row"]]
        parameter = PARAM_NAMES[error["jacobian_col"]]
        rms = np.sqrt(np.mean(jacobian[f"{method}_rms_abs_error"] ** 2))
        print(f"\n{method.upper()}")
        print(f"  Maximum absolute error: {error['max_abs']:.6e}")
        print(f"  RMS absolute error:     {rms:.6e}")
        print(f"  Maximum relative error: {jacobian[f'{method}_max_rel_error'].max():.6e}")
        print(f"  Worst interval:         {error['MD_start']:.3f} -> {error['MD_end']:.3f} m")
        print(f"  Worst entry:            d({coordinate})/d({parameter})")
    print("\nFINITE-DIFFERENCE STEP-SIZE CONVERGENCE")
    if not convergence.empty:
        columns = ["leg_index", "step_multiplier", "mcm_max_abs_error", "bt_max_abs_error"]
        print(convergence[columns].to_string(index=False, float_format="{:.6e}".format))


def run_validation(csv_path, fd_steps=None, step_multipliers=(10, 1, 0.1, 0.01, 0.001),
                   representative_leg_indices=None):
    """Run the report's FD checks and return the data used by Figure 3."""
    if fd_steps is None:
        fd_steps = DEFAULT_FD_STEPS
    survey = load_well(csv_path)
    print(f"\nSurvey: {csv_path} | Physical stations: {len(survey)}")
    print(f"FD steps: inclination {fd_steps['inc']:g} deg, azimuth {fd_steps['azi']:g} deg, "
          f"length {fd_steps['dl']:g} m")
    jacobian, worst = validate_jacobians_over_well(survey, fd_steps)
    if representative_leg_indices is None:
        n = len(survey) - 1
        representative_leg_indices = sorted({1, max(1, n // 3), max(1, 2 * n // 3), n})
    convergence = run_fd_convergence_tests(
        survey, step_multipliers, representative_leg_indices, base_steps=fd_steps,
    )
    print_summary(jacobian, convergence, worst)
    return jacobian, convergence


if __name__ == "__main__":
    run_validation(
        csv_path="data/15_9_F_11_A.csv",
        fd_steps={"inc": 1e-3, "azi": 1e-3, "dl": 0.1},
        step_multipliers=(10, 1, 0.1, 0.01, 0.001),
    )
