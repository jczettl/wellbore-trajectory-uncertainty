import numpy as np
import pandas as pd
from pathlib import Path

from core_math import (
    get_confidence_k_factor,
    get_ellipse_parameters,
    get_jacobian,
)


# =====================================================================
# JACOBIAN COMPARISON REPORTING
# =====================================================================

_JACOBIAN_COLUMN_METADATA = (
    ("I1", "Start inclination", "m/deg", r"$I_1$", r"$\mathrm{m/deg}$"),
    ("A1", "Start azimuth", "m/deg", r"$A_1$", r"$\mathrm{m/deg}$"),
    ("I2", "End inclination", "m/deg", r"$I_2$", r"$\mathrm{m/deg}$"),
    ("A2", "End azimuth", "m/deg", r"$A_2$", r"$\mathrm{m/deg}$"),
    ("delta_MD", "Course length", "m/m", r"$\Delta MD$", r"$\mathrm{m/m}$"),
)


def _latex_scientific(value):
    """Format a finite scalar for a compact LaTeX numerical table."""
    value = float(value)
    if not np.isfinite(value):
        return r"---"
    if np.isclose(value, 0.0, rtol=0.0, atol=1.0e-300):
        return r"$0$"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return r"${:.3f}\times10^{{{}}}$".format(mantissa, exponent)


def _write_jacobian_comparison_latex(summary, output_path):
    """Write a complete LaTeX table that can be included in the report."""
    metadata_by_key = {
        key: (latex_label, latex_unit)
        for key, _, _, latex_label, latex_unit in _JACOBIAN_COLUMN_METADATA
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Column-wise comparison of analytical Minimum Curvature "
            r"and Balanced Tangential segment sensitivities. Each absolute "
            r"statistic is calculated from the Euclidean norm of the NEV "
            r"difference vector for one parameter column.}"
        ),
        r"\label{tab:mcm_bt_jacobian_columns}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l c r r r r r@{}}",
        r"\toprule",
        (
            r"\textbf{Column} & \textbf{Unit} & \textbf{RMS} & "
            r"\textbf{95th pct.} & \textbf{Maximum} & "
            r"\textbf{Relative RMS} & \textbf{MD at max. (m)} \\"
        ),
        r"\midrule",
    ]

    for row in summary.itertuples(index=False):
        latex_label, latex_unit = metadata_by_key[row.parameter]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {:.2f} \\\\".format(
                latex_label,
                latex_unit,
                _latex_scientific(row.rms_difference),
                _latex_scientific(row.percentile_95_difference),
                _latex_scientific(row.maximum_difference),
                _latex_scientific(row.relative_rms),
                row.md_at_maximum_m,
            )
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def generate_jacobian_comparison_table(
    df,
    output_dir="jacobian_comparison_results",
    file_stem="mcm_vs_bt_jacobian_columns",
    print_table=True,
    md_col="MD",
    inc_col="Incl",
    azi_col="Azi",
):
    """Summarize MCM-versus-BT Jacobian differences by parameter column.

    Only physical survey stations are used when ``is_interpolated`` is present.
    For leg ``k`` and parameter column ``j``, the absolute sample is

        d[k, j] = ||J_MCM[k, :, j] - J_BT[k, :, j]||_2.

    The relative RMS is the root-sum-square difference divided by the
    root-sum-square BT reference column over all physical survey legs. This
    preserves the native units of each column and avoids mixing angular and
    course-length sensitivities in one Frobenius statistic.

    Args:
        df (pd.DataFrame): Survey or calculated trajectory dataframe.
        output_dir (str or pathlib.Path): Directory for CSV and LaTeX output.
            Pass ``None`` to disable file output.
        file_stem (str): Shared filename stem for the output files.
        print_table (bool): Print a compact human-readable table to stdout.
        md_col (str): Measured-depth column.
        inc_col (str): Inclination column in degrees.
        azi_col (str): Azimuth column in degrees.

    Returns:
        pd.DataFrame: One row per Jacobian parameter column.
    """
    required_columns = [md_col, inc_col, azi_col]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Cannot generate Jacobian comparison table; missing columns: "
            + ", ".join(missing)
        )

    if "is_interpolated" in df.columns:
        physical = df.loc[~df["is_interpolated"].astype(bool)].copy()
    else:
        physical = df.copy()
    physical = physical.reset_index(drop=True)

    if len(physical) < 2:
        raise ValueError(
            "At least two physical survey stations are required for the "
            "Jacobian comparison table."
        )

    values = physical[required_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(
            "Jacobian comparison input contains non-numeric or non-finite values."
        )

    md = values[md_col].to_numpy(dtype=float)
    inc = values[inc_col].to_numpy(dtype=float)
    azi = values[azi_col].to_numpy(dtype=float)
    delta_md = np.diff(md)
    if np.any(delta_md <= 0.0):
        raise ValueError(
            "Physical survey MD must be strictly increasing for the Jacobian "
            "comparison table."
        )

    number_of_legs = len(physical) - 1
    mcm_jacobians = np.empty((number_of_legs, 3, 5), dtype=float)
    bt_jacobians = np.empty_like(mcm_jacobians)

    for leg in range(number_of_legs):
        args = (
            inc[leg],
            inc[leg + 1],
            azi[leg],
            azi[leg + 1],
            delta_md[leg],
        )
        mcm_jacobians[leg] = get_jacobian(
            *args,
            balanced_tangential_jacobi_approximation=False,
        )
        bt_jacobians[leg] = get_jacobian(
            *args,
            balanced_tangential_jacobi_approximation=True,
        )

    if not (
        np.isfinite(mcm_jacobians).all()
        and np.isfinite(bt_jacobians).all()
    ):
        raise ValueError("Jacobian comparison produced non-finite values.")

    difference_norms = np.linalg.norm(
        mcm_jacobians - bt_jacobians,
        axis=1,
    )
    bt_column_norms = np.linalg.norm(bt_jacobians, axis=1)
    rows = []

    for column, metadata in enumerate(_JACOBIAN_COLUMN_METADATA):
        key, description, unit, _, _ = metadata
        samples = difference_norms[:, column]
        maximum_leg = int(np.argmax(samples))
        reference_rss = float(np.linalg.norm(bt_column_norms[:, column]))
        difference_rss = float(np.linalg.norm(samples))

        if reference_rss > np.finfo(float).eps:
            relative_rms = difference_rss / reference_rss
        elif difference_rss <= np.finfo(float).eps:
            relative_rms = 0.0
        else:
            relative_rms = np.nan

        rows.append(
            {
                "parameter": key,
                "description": description,
                "unit": unit,
                "rms_difference": float(np.sqrt(np.mean(samples ** 2))),
                "percentile_95_difference": float(np.percentile(samples, 95.0)),
                "maximum_difference": float(samples[maximum_leg]),
                "relative_rms": relative_rms,
                "relative_rms_percent": 100.0 * relative_rms,
                "md_at_maximum_m": float(md[maximum_leg + 1]),
            }
        )

    summary = pd.DataFrame(rows)

    if print_table:
        display = summary[
            [
                "parameter",
                "unit",
                "rms_difference",
                "percentile_95_difference",
                "maximum_difference",
                "relative_rms_percent",
                "md_at_maximum_m",
            ]
        ].copy()
        display.columns = [
            "Column",
            "Unit",
            "RMS difference",
            "95th percentile",
            "Maximum",
            "Relative RMS (%)",
            "MD at maximum (m)",
        ]
        print("\n" + "=" * 100)
        print("MCM VERSUS BALANCED TANGENTIAL JACOBIAN COLUMNS")
        print("=" * 100)
        print(
            display.to_string(
                index=False,
                formatters={
                    "RMS difference": "{:.6e}".format,
                    "95th percentile": "{:.6e}".format,
                    "Maximum": "{:.6e}".format,
                    "Relative RMS (%)": "{:.6e}".format,
                    "MD at maximum (m)": "{:.2f}".format,
                },
            )
        )
        print(
            "Absolute statistics use the NEV vector norm for each column; "
            "relative RMS uses BT as the reference."
        )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / (file_stem + ".csv")
        latex_path = output_path / (file_stem + ".tex")
        summary.to_csv(csv_path, index=False)
        _write_jacobian_comparison_latex(summary, latex_path)
        print("Jacobian comparison CSV: {}".format(csv_path.resolve()))
        print("Jacobian comparison LaTeX: {}".format(latex_path.resolve()))

    return summary

# =====================================================================
# FINAL SURVEY (END-OF-WELL) REPORT GENERATION
# =====================================================================

def generate_eow_report(
    df,
    C_total_array=None,
    vs_azimuth=None,
    dls_interval=30.0,
    confidence=0.95,
    scale_mode='sigma',
    sigma_level=2.0,
    **kwargs,
):
    """
    Generates an expanded, professional Final Survey (End-Of-Well) Report.
    
    This function compiles comprehensive wellbore geometry telemetry, trajectory 
    statistics, extreme dogleg metrics, and terminal uncertainty ellipses. It accepts 
    metadata keywords dynamically via **kwargs to format custom corporate header profiles.

    Args:
        df (pd.DataFrame): Trajectory dataframe containing depth, inclination, and azimuth data.
        C_total_array (np.ndarray, optional): Array of 3x3 covariance matrices across stations. 
                                             Defaults to None.
        vs_azimuth (float, optional): Target azimuth heading for Vertical Section calculation. 
                                     Defaults to None (uses closure azimuth).
        dls_interval (float): Standardization interval length for Dogleg Severity normalization. Defaults to 30.0.
        confidence (float): Joint coverage probability in probability mode.
        scale_mode (str): ``'sigma'`` (default) or ``'probability'``.
        sigma_level (float): Axis multiplier in sigma mode. Defaults to 2.
        **kwargs: Flexible metadata parameters (operator, well, field, site, api_uwi, geodetic_datum, etc.).

    Returns:
        pd.DataFrame: The filtered actual survey stations DataFrame with appended Dogleg and Vertical Section metrics.
    """
    # Step 1: Extract metadata fields with safe fallback defaults if they aren't provided in kwargs
    operator       = kwargs.get("operator", "N/A")
    well           = kwargs.get("well", "N/A")
    field          = kwargs.get("field", "N/A")
    site           = kwargs.get("site", "N/A")
    country_state  = kwargs.get("country_state", "N/A")
    api_uwi        = kwargs.get("api_uwi", "N/A")
    geodetic_datum = kwargs.get("geodetic_datum", "N/A")
    north_ref      = kwargs.get("north_ref", "N/A")
    system_grid    = kwargs.get("system_grid", "N/A")
    convergence    = kwargs.get("convergence", 0.0)
    declination    = kwargs.get("declination", 0.0)
    elevation_ref  = kwargs.get("elevation_ref", "N/A")
    measured_from  = kwargs.get("measured_from", "N/A")
    survey_method  = kwargs.get("survey_method", "Minimum Curvature")
    error_model    = kwargs.get("error_model", "ISCWSA MWD Rev.5.1 MINI")

    # Step 2: Filter out artificially interpolated structural grid points to focus strictly on actual survey stations
    if 'is_interpolated' in df.columns:
        df_actual = df[~df['is_interpolated']].copy().reset_index(drop=True)
    else:
        df_actual = df.copy()
        
    n_stations = len(df_actual)
    
    # Step 3: Extract extreme survey milestones, dogleg metrics, and closure statistics
    max_dls_idx = df_actual['DLS'].idxmax() if n_stations > 1 else 0
    max_dls_val = df_actual['DLS'].max() if n_stations > 1 else 0.0
    max_dls_md = df_actual.loc[max_dls_idx, 'MD'] if n_stations > 1 else df_actual['MD'].iloc[0]
    avg_dls_val = df_actual['DLS'].mean() if n_stations > 1 else 0.0

    first = df_actual.iloc[0]
    bhl = df_actual.iloc[-1]

    delta_ns = bhl['NS'] - first['NS']
    delta_ew = bhl['EW'] - first['EW']

    closure_dist = np.sqrt(delta_ns**2 + delta_ew**2)
    
    # Step 4: Compute Vertical Section (VS) projections along the specified target azimuth heading
    if np.isclose(closure_dist, 0.0):
        closure_azi = 0.0
    else:
        closure_azi = (
            np.degrees(np.arctan2(delta_ew, delta_ns)) + 360.0
        ) % 360.0

    if vs_azimuth is None:
        vs_azimuth = closure_azi

    vs_azi_rad = np.radians(vs_azimuth)

    ns_offset = df_actual['NS'] - first['NS']
    ew_offset = df_actual['EW'] - first['EW']

    df_actual['VS'] = ns_offset * np.cos(vs_azi_rad) + ew_offset * np.sin(vs_azi_rad)
    final_vs = df_actual['VS'].iloc[-1]
    
    # Step 5: Extract Ellipsoid of Uncertainty (EOU) parameters at Total Depth using decomposition helpers
    semi_major, semi_minor, angle, vertical_unc = 0.0, 0.0, 0.0, 0.0
    confidence_description = "Not calculated"
    
    if C_total_array is not None:
        # Extract terminal covariance matrix slice for bottom-hole location (BHL)
        C_bhl_array = np.expand_dims(C_total_array[-1], axis=0)
        
        # Leverage ellipse parameter decomposition to find semi-axes and orientation angles
        semi_majors, semi_minors, angles = get_ellipse_parameters(
            C_bhl_array,
            plane='NE',
            confidence=confidence,
            scale_mode=scale_mode,
            sigma_level=sigma_level,
        )
        semi_major, semi_minor, angle = semi_majors[0], semi_minors[0], np.degrees(angles[0])
        
        # Extract vertical depth uncertainty component (Z-axis variance profile)
        k_factor_1d = get_confidence_k_factor(
            confidence=confidence,
            dimensions=1,
            scale_mode=scale_mode,
            sigma_level=sigma_level,
        )
        vertical_unc = k_factor_1d * np.sqrt(np.clip(C_total_array[-1, 2, 2], 0, None))

        k_factor_2d = get_confidence_k_factor(
            confidence=confidence,
            dimensions=2,
            scale_mode=scale_mode,
            sigma_level=sigma_level,
        )
        if str(scale_mode).lower() == 'sigma':
            confidence_description = f"{sigma_level:g}σ EOU (k={k_factor_2d:.4f})"
        else:
            confidence_description = (
                f"{confidence * 100:g}% joint NE ellipse "
                f"(k₂={k_factor_2d:.4f}); {confidence * 100:g}% "
                f"vertical marginal (z={k_factor_1d:.4f})"
            )

    ref_short = measured_from.split(' ')[0] if ' ' in measured_from else measured_from
    tie_in_type = "Surface Origin" if np.isclose(df_actual['MD'].iloc[0], 0.0) else "Subsurface Tie-In"

    # ---------------------------------------------------------------------
    # RENDER ENGINE (Strict 85-Character Aligned Layout)
    # ---------------------------------------------------------------------
    print("=" * 85)
    print("FINAL DIRECTIONAL SURVEY REPORT".center(85))
    print("=" * 85)
    print(f"{'OPERATOR:':<16}{operator:<34}  {'WELL:':<14}{well}")
    print(f"{'FIELD:':<16}{field:<34}  {'SITE:':<14}{site}")
    print(f"{'COUNTRY/STATE:':<16}{country_state:<34}  {'API/UWI:':<14}{api_uwi}")
    print("-" * 85)
    print(f"{'GEODETIC DATUM:':<16}{geodetic_datum:<34}  {'NORTH REF:':<14}{north_ref}")
    print(f"{'SYSTEM GRID:':<16}{system_grid:<34}  {'CONVERGENCE:':<14}{convergence:+.2f}°")
    print(f"{'DECLINATION:':<16}{declination:+.2f}°")
    print(f"{'ELEVATION REF:':<16}{elevation_ref:<34}  {'MEASURED FROM:':<14}{measured_from}")
    print("-" * 85)
    print(f"{'SURVEY METHOD:':<16}{survey_method:<34}  {'ERROR MODEL:':<14}{error_model}")
    print(f"{'TIE-IN STATION:':<16}{tie_in_type} ({df_actual['MD'].iloc[0]:.2f} MD, {df_actual['Incl'].iloc[0]:.2f} Inc, {df_actual['Azi'].iloc[0]:.2f} Azi)")
    print("=" * 85)
    print("SURVEY TRAJECTORY SUMMARY".center(85))
    print("=" * 85)
    print(f"{'Total Survey Stations:':<32}{n_stations}")
    print(f"{'Total Measured Depth:':<32}{bhl['MD']:.2f} m")
    print(f"{'True Vertical Depth:':<32}{bhl['TVD']:.2f} m (Ref: {ref_short})")
    print("-" * 85)
    print(f"{'Final Inclination:':<32}{bhl['Incl']:.2f}°")
    print(f"{'Final Azimuth:':<32}{bhl['Azi']:.2f}°")
    print(f"{'Bottom Hole North (Y):':<32}{bhl['NS']:.2f} m")
    print(f"{'Bottom Hole East (X):':<32}{bhl['EW']:.2f} m")
    print("-" * 85)
    print(f"{'Closure Distance:':<32}{closure_dist:.2f} m")
    print(f"{'Closure Azimuth:':<32}{closure_azi:.2f}°")
    print(f"{'Vertical Section:':<32}{final_vs:.2f} m (Projected @ {vs_azimuth:.2f}°)")
    print("-" * 85)
    print(f"{'Maximum Dogleg Severity:':<32}{max_dls_val:.3f} °/{int(dls_interval)}m  @ {max_dls_md:.2f} m MD")
    print("=" * 85)
    print("POSITION UNCERTAINTY AT TD".center(85))
    print("=" * 85)
    print(
        f"{'Confidence Level:':<32}"
        f"{confidence_description}"
    )
    print(f"{'Horizontal Semi-Major:':<32}{semi_major:.2f} m")
    print(f"{'Horizontal Semi-Minor:':<32}{semi_minor:.2f} m")
    print(f"{'Major-Axis Azimuth:':<32}{angle:.2f}°")
    print(f"{'Vertical Marginal Interval:':<32}±{vertical_unc:.2f} m")
    print("=" * 85)
    
    return df_actual
