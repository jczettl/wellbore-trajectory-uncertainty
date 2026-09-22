import pandas as pd
import numpy as np

# =====================================================================
# CONTEXTUAL DEBUGGING ABSTRACTIONS
# =====================================================================

def _get_row_context(df, idx):
    """
    Helper function to fetch the error row alongside the preceding and succeeding rows.
    Provides immediate visual context for terminal debugging during QC failures.

    Args:
        df (pd.DataFrame): The trajectory dataframe being evaluated.
        idx (int): The index of the row where the error occurred.

    Returns:
        str: A formatted string block of the dataframe context.
    """
    start = max(0, idx - 1)
    end = min(len(df), idx + 2)
    return "\n" + df.iloc[start:end].to_string() + "\n"

def _raise_qc_error(df, bad_mask, error_msg):
    """
    Centralized hard-halt error handling function. 
    
    Evaluates a boolean mask, locates the first failure point, fetches the data 
    context, and explicitly halts execution to prevent poisoned data from 
    entering the mathematical engine.

    Args:
        df (pd.DataFrame): The trajectory dataframe.
        bad_mask (pd.Series): Boolean mask where True indicates a physical violation.
        error_msg (str): Custom description of the specific QC failure.
        
    Raises:
        ValueError: Explicit halt containing the error message and dataframe context.
    """
    if bad_mask.any():
        first_bad_idx = df[bad_mask].index[0]
        context = _get_row_context(df, first_bad_idx)
        raise ValueError(f"QC FAILED: {error_msg} Data context:{context}")


# =====================================================================
# CORE QUALITY CONTROL GATE
# =====================================================================

def validate_and_clean_survey(
    df_input,
    md_col='MD',
    inc_col='Incl',
    azi_col='Azi',
    detailed_warnings=False,
    min_md_interval=1e-6,
    reject_short_intervals=False,
):
    """
    Primary Quality Control (QC) Gate for wellbore survey telemetry. 
    
    This function enforces strict physical limits and mathematical safety prior to 
    spatial propagation. It rejects non-finite values and conflicting duplicate
    stations, normalizes azimuth, and reports unusually short intervals.

    Args:
        df_input (pd.DataFrame): Raw trajectory dataframe.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        inc_col (str): Column name for Inclination. Defaults to 'Incl'.
        azi_col (str): Column name for Azimuth. Defaults to 'Azi'.
        detailed_warnings (bool): If True, prints row-by-row context for every 
                                  short interval. Defaults to False.
        min_md_interval (float): Positive interval below which a row is flagged.
        reject_short_intervals (bool): If True, flagged positive intervals fail QC.

    Returns:
        pd.DataFrame: A mathematically safe, floating-point trajectory dataframe ready for calculation.
    """
    df = df_input.copy().reset_index(drop=True)
    required_cols = [md_col, inc_col, azi_col]
    
    # Ensure required telemetry columns exist before proceeding
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"QC FAILED: Missing required column '{col}' in dataset.")

    if df.empty:
        raise ValueError("QC FAILED: Survey contains no stations.")

    # 1. Replace empty strings or whitespace with actual NaNs for uniform programmatic detection
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # 2. Hard Error for Missing Values / NaNs (Prevents matrix propagation collapse)
    for col in required_cols:
        _raise_qc_error(df, df[col].isna(), f"Missing value (NaN) detected in column '{col}'.")

    # Convert once, then check again: strings such as "nan" are not missing until
    # after numeric conversion.
    df[required_cols] = df[required_cols].astype(float)
    for col in required_cols:
        _raise_qc_error(
            df,
            ~np.isfinite(df[col]),
            f"Non-finite value detected in column '{col}'.",
        )

    if not np.isfinite(min_md_interval) or min_md_interval < 0:
        raise ValueError("min_md_interval must be a finite non-negative number.")

    # 3. First Line Check (Subsurface Tie-in Handling)
    if not np.isclose(df[md_col].iloc[0], 0.0):
        first_md = df[md_col].iloc[0]
        print(f"  | QC WARNING: First MD is {first_md}m (not 0.0m).")
        print(f"  |             Treating this as a Subsurface Tie-In. All calculated North/East coordinates")
        print(f"  |             will be relative to this starting point unless absolute Tie-In coordinates are provided.")

    # 4. Negative MD Check (Physically impossible to drill upwards past the surface datum)
    _raise_qc_error(df, df[md_col] < 0, "Negative Measured Depth (MD) detected.")

    # 5. Out-of-order values are never sorted automatically because that can hide
    # a source-data problem.
    _raise_qc_error(df, df[md_col].diff() < 0, "Measured Depth (MD) values are not strictly increasing (out of order).")

    # 6. Inclination Limits Check (Physically restricted to the 0-180 degree downward domain)
    _raise_qc_error(df, (df[inc_col] < 0) | (df[inc_col] > 180), "Inclination out of physical bounds (0-180°).")

    # 7. Azimuth Normalization (Modulo 360 wrap-around handling)
    bad_azi_mask = (df[azi_col] < 0) | (df[azi_col] >= 360)
    if bad_azi_mask.any():
        bad_azi_indices = df[bad_azi_mask].index.tolist()
        for idx in bad_azi_indices:
            orig_azi = df.loc[idx, azi_col]
            # Snap out-of-bounds azimuths cleanly onto the 360-degree compass rose
            new_azi = orig_azi % 360
            df.loc[idx, azi_col] = new_azi
            
            print(f"  | QC WARNING: Azimuth out of bounds at MD {df.loc[idx, md_col]} m. Original: {orig_azi:.2f}°, Corrected modulo 360: {new_azi:.2f}°")

    # 8. Duplicate MDs with different attitudes are conflicting measurements and
    # must not be silently deleted. Exact repeated records are harmless and are
    # collapsed to one row with an explicit message.
    duplicate_mask = df[md_col].diff().eq(0.0)
    duplicate_rows_to_drop = []
    for idx in df.index[duplicate_mask]:
        same_inc = np.isclose(df.loc[idx, inc_col], df.loc[idx - 1, inc_col])
        same_azi = np.isclose(df.loc[idx, azi_col], df.loc[idx - 1, azi_col])
        if not (same_inc and same_azi):
            conflict_mask = pd.Series(False, index=df.index)
            conflict_mask.loc[idx] = True
            _raise_qc_error(
                df,
                conflict_mask,
                "Conflicting survey attitudes found at the same measured depth.",
            )
        duplicate_rows_to_drop.append(idx)

    if duplicate_rows_to_drop:
        df = df.drop(index=duplicate_rows_to_drop).reset_index(drop=True)
        print(
            f"  | QC INFO: Collapsed {len(duplicate_rows_to_drop)} exact duplicate "
            "survey row(s)."
        )

    # 9. Positive short intervals are retained by default. Projects that prohibit
    # them can opt into a hard failure without changing the numerical code.
    md_diffs = df[md_col].diff()
    short_mask = (md_diffs > 0.0) & (md_diffs < min_md_interval)
    if short_mask.any():
        if reject_short_intervals:
            _raise_qc_error(
                df,
                short_mask,
                f"Measured-depth interval below {min_md_interval:g} m.",
            )
        print(
            f"  | QC WARNING: {int(short_mask.sum())} positive MD interval(s) "
            f"are below {min_md_interval:g} m and were retained."
        )
        if detailed_warnings:
            for idx in df.index[short_mask]:
                print(_get_row_context(df, idx))

    print("  | QC PASS: Survey data is mathematically safe and within physical bounds.")
    return df
