import numpy as np
import pandas as pd
import os

# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================
# Module-level constant defining covariance matrix indices to avoid redundant 
# mapping dictionaries during heavy numerical iterations.
_COMPONENT_MAP = {
    "NN": (0, 0), "EE": (1, 1), "VV": (2, 2), 
    "NE": (0, 1), "NV": (0, 2), "EV": (1, 2)
}

# =====================================================================
# NUMERICAL VALIDATION & ERROR QUANTIFICATION
# =====================================================================

def validate_tvd(df_out, wellpath_data, tolerance=1e-6, md_tolerance=1e-8):
    """
    Compares calculated True Vertical Depth (TVD) against reference wellpath TVD values.

    This function isolates vertical integration drift by subtracting the calculated values
    from the Excel-provided benchmark dataset, logging any deviations that breach the 
    predefined physical tolerance.

    Args:
        df_out (pd.DataFrame): DataFrame containing calculated spatial trajectory positions.
        wellpath_data (pd.DataFrame): Raw wellpath reference dataset extracted from Excel.
        tolerance (float): Acceptable depth deviation limit in meters. Defaults to 10^-6 (std of 1mm).

    Returns:
        tuple: A pair containing (mismatches DataFrame, max_tvd_diff, max_tvd_pct).
    """
    if len(df_out) != len(wellpath_data):
        raise ValueError(
            f"TVD benchmark length mismatch: calculated={len(df_out)}, "
            f"reference={len(wellpath_data)}."
        )

    required = {'MD', 'TVD'}
    missing_calc = required.difference(df_out.columns)
    missing_ref = required.difference(wellpath_data.columns)
    if missing_calc or missing_ref:
        raise ValueError(
            f"TVD benchmark missing columns; calculated={sorted(missing_calc)}, "
            f"reference={sorted(missing_ref)}."
        )

    calc_values = df_out[['MD', 'TVD']].to_numpy(dtype=float)
    ref_values = wellpath_data[['MD', 'TVD']].to_numpy(dtype=float)
    if not np.isfinite(calc_values).all() or not np.isfinite(ref_values).all():
        raise ValueError("TVD benchmark contains non-finite MD or TVD values.")
    if not np.allclose(calc_values[:, 0], ref_values[:, 0], rtol=0.0, atol=md_tolerance):
        max_md_diff = float(np.max(np.abs(calc_values[:, 0] - ref_values[:, 0])))
        raise ValueError(
            f"TVD benchmark MD alignment failed; maximum difference={max_md_diff:.3e}."
        )

    df_out['Original_TVD'] = ref_values[:, 1]
    df_out['TVD_Diff'] = (df_out['TVD'] - df_out['Original_TVD']).abs()
    
    # Calculate percentage error safely while explicitly handling zero divisions at the surface
    df_out['TVD_Pct_Err'] = np.where(
        df_out['Original_TVD'] != 0,
        (df_out['TVD_Diff'] / df_out['Original_TVD'].abs()) * 100,
        np.where(df_out['TVD_Diff'] == 0, 0.0, np.inf)
    )
    
    # Extract max error bounds and filter mismatches that breach the operational tolerance
    max_tvd_diff = df_out['TVD_Diff'].max()
    max_tvd_pct = df_out.loc[df_out['TVD_Pct_Err'] != np.inf, 'TVD_Pct_Err'].max()
    mismatches = df_out[df_out['TVD_Diff'] > tolerance]
    
    return mismatches, max_tvd_diff, max_tvd_pct

def compare_covariances(
    df_out,
    calc_covs,
    excel_covs,
    tolerance=1e-6,
    md_tolerance=1e-8,
):
    """
    Numerically compares calculated covariance matrices with reference Excel benchmark tables.

    Compares the six independent covariance components (NN, EE, VV, NE, NV,
    EV) at every aligned survey station.

    Args:
        df_out (pd.DataFrame): Trajectory dataframe containing Measured Depths.
        calc_covs (dict): Dictionary of calculated covariance arrays.
        excel_covs (dict): Dictionary of benchmark Excel covariance DataFrames.
        tolerance (float): Numerical error tolerance threshold. Defaults to 10^-6 (std of 1mm).

    Returns:
        dict: Summary dictionary containing passed terms, failed term error logs, and global maximum differences.
    """
    if not calc_covs:
        raise ValueError("No calculated covariance terms were supplied.")

    expected_terms = set(calc_covs)
    reference_terms = set(excel_covs)
    missing_reference_terms = sorted(expected_terms - reference_terms)
    if missing_reference_terms:
        raise ValueError(
            "Missing reference covariance sheets for calculated terms: "
            + ", ".join(missing_reference_terms)
        )

    required_reference_columns = {'MD', *_COMPONENT_MAP.keys()}
    results = {
        'passed_terms': [],
        'failed_terms': {},
        'global_max_diff': 0.0,
        'expected_terms': sorted(expected_terms),
        'compared_terms': [],
    }
    calculated_md = df_out['MD'].to_numpy(dtype=float)
    if not np.isfinite(calculated_md).all():
        raise ValueError("Calculated benchmark MD values contain non-finite entries.")
    
    for term in calc_covs:
        calc_matrix_array = calc_covs[term]  
        df_excel = excel_covs[term]

        missing_columns = required_reference_columns.difference(df_excel.columns)
        if missing_columns:
            raise ValueError(
                f"Reference term '{term}' is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        if calc_matrix_array.shape != (len(df_out), 3, 3):
            raise ValueError(
                f"Calculated term '{term}' has shape {calc_matrix_array.shape}; "
                f"expected {(len(df_out), 3, 3)}."
            )
        if len(df_excel) != len(df_out):
            raise ValueError(
                f"Covariance benchmark length mismatch for '{term}': "
                f"calculated={len(df_out)}, reference={len(df_excel)}."
            )

        reference_values = df_excel[['MD', *_COMPONENT_MAP.keys()]].to_numpy(dtype=float)
        if not np.isfinite(calc_matrix_array).all() or not np.isfinite(reference_values).all():
            raise ValueError(f"Covariance benchmark term '{term}' contains non-finite values.")
        if not np.allclose(calculated_md, reference_values[:, 0], rtol=0.0, atol=md_tolerance):
            max_md_diff = float(np.max(np.abs(calculated_md - reference_values[:, 0])))
            raise ValueError(
                f"Covariance benchmark MD alignment failed for '{term}'; "
                f"maximum difference={max_md_diff:.3e}."
            )

        results['compared_terms'].append(term)
        
        mismatches = []
        max_term_diff = 0.0
        max_term_pct = 0.0
        
        def calc_pct(c_val, e_val):
            if e_val != 0: return (abs(c_val - e_val) / abs(e_val)) * 100
            return 0.0 if c_val == 0 else float('inf')
        
        # Step through each survey station node to cross-validate matrix components
        for i in range(len(df_out)):
            md = df_out.iloc[i]['MD']
            
            # Extract calculated components from NumPy 3x3 matrix
            NN_c, EE_c, VV_c = calc_matrix_array[i, 0, 0], calc_matrix_array[i, 1, 1], calc_matrix_array[i, 2, 2]
            NE_c, NV_c, EV_c = calc_matrix_array[i, 0, 1], calc_matrix_array[i, 0, 2], calc_matrix_array[i, 1, 2] 
            
            # Extract benchmark components from Pandas DataFrame
            row_excel = df_excel.iloc[i]
            NN_e, EE_e, VV_e = row_excel['NN'], row_excel['EE'], row_excel['VV']
            NE_e, NV_e, EV_e = row_excel['NE'], row_excel['NV'], row_excel['EV']
            
            diffs = {
                'NN': abs(NN_c - NN_e), 'EE': abs(EE_c - EE_e), 'VV': abs(VV_c - VV_e),
                'NE': abs(NE_c - NE_e), 'NV': abs(NV_c - NV_e), 'EV': abs(EV_c - EV_e)
            }
            pct_errs = {
                'NN': calc_pct(NN_c, NN_e), 'EE': calc_pct(EE_c, EE_e), 'VV': calc_pct(VV_c, VV_e),
                'NE': calc_pct(NE_c, NE_e), 'NV': calc_pct(NV_c, NV_e), 'EV': calc_pct(EV_c, EV_e)
            }
            
            # Track maximum mathematical limits for reporting
            current_max_diff = max(diffs.values())
            valid_pcts = [p for p in pct_errs.values() if p != float('inf')]
            current_max_pct = max(valid_pcts) if valid_pcts else 0.0
            
            max_term_diff = max(max_term_diff, current_max_diff)
            max_term_pct = max(max_term_pct, current_max_pct)
            
            # Log mismatch if the numerical tolerance is breached
            max_diff_label = max(diffs, key=diffs.get)
            if diffs[max_diff_label] > tolerance:
                mismatches.append({
                    'MD': md, 'Worst_Component': max_diff_label,
                    'Error_Magnitude': diffs[max_diff_label], 'Error_Pct': pct_errs[max_diff_label]
                })
                
        results['global_max_diff'] = max(results['global_max_diff'], max_term_diff)
        
        if not mismatches:
            results['passed_terms'].append(term)
        else:
            results['failed_terms'][term] = {
                'mismatches': mismatches, 'max_diff': max_term_diff, 'max_pct': max_term_pct
            }

    return results

def covariance_rms_relative_error(calc_covs, excel_covs, components=("NN","EE","VV", "NE", "NV", "EV"), zero_tol=1e-12):
    """
    Computes the Root Mean Square (RMS) relative error for every ISCWSA error term.

    Args:
        calc_covs (dict): Dictionary of calculated covariance arrays.
        excel_covs (dict): Dictionary of reference Excel covariance DataFrames.
        components (tuple): Component string keys to evaluate. Defaults to the 6 standard components.
        zero_tol (float): Numerical threshold to treat truth values as zero. Defaults to 1e-12.

    Returns:
        pd.DataFrame: A dataframe containing RMS relative error metrics per term, sorted by overall RMS.
    """
    rows = []

    for term in sorted(calc_covs.keys()):
        if term not in excel_covs: continue
        row = {"Error Term": term}
        total = []

        # Iterate using the centralized module-level component map
        for comp in components:
            i, j = _COMPONENT_MAP[comp]
            calc = calc_covs[term][:, i, j]
            truth = excel_covs[term][comp].to_numpy()
            
            # Safely align array lengths
            n = min(len(calc), len(truth))
            calc, truth = calc[:n], truth[:n]

            # Calculate relative error avoiding division by zero via masking
            mask = np.abs(truth) > zero_tol
            if np.any(mask):
                rel = (truth[mask] - calc[mask]) / truth[mask]
                row[comp] = np.sqrt(np.mean(rel**2))
                total.extend(rel**2)
            else:
                row[comp] = np.nan

        row["Overall RMS"] = np.sqrt(np.mean(total)) if total else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Overall RMS")

def covariance_max_absolute_error(calc_covs, excel_covs, components=("NN", "EE", "VV", "NE", "NV", "EV")):
    """
    Computes the maximum absolute difference for every ISCWSA error term.

    Args:
        calc_covs (dict): Dictionary of calculated covariance arrays.
        excel_covs (dict): Dictionary of reference Excel covariance DataFrames.
        components (tuple): Component string keys to evaluate. Defaults to standard 6 components.

    Returns:
        pd.DataFrame: A dataframe containing maximum absolute differences per term, sorted descending.
    """
    rows = []

    for term in sorted(calc_covs.keys()):
        if term not in excel_covs: continue
        row = {"Error Term": term}
        max_overall = 0.0

        # Iterate using the centralized module-level component map
        for comp in components:
            i, j = _COMPONENT_MAP[comp]
            
            calc = calc_covs[term][:, i, j]
            truth = excel_covs[term][comp].to_numpy()
            
            # Safely align array lengths
            n = min(len(calc), len(truth))
            calc, truth = calc[:n], truth[:n]

            # Calculate raw absolute difference
            abs_diff = np.abs(calc - truth)
            max_diff = np.max(abs_diff) if len(abs_diff) > 0 else np.nan
            row[comp] = max_diff
            
            if not np.isnan(max_diff) and max_diff > max_overall:
                max_overall = max_diff

        row["Overall Max Diff"] = max_overall
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Overall Max Diff", ascending=False)


# =====================================================================
# BATCH TRAJECTORY ANALYTICS & EXPORT
# =====================================================================

def export_tvd_data(file_paths, pipeline_runner, output_suffix="_tvd_investigation.csv"):
    """
    Executes a batch processing pipeline across multiple wellbore datasets to extract 
    and compile True Vertical Depth (TVD) deviation metrics.
    
    This analytical tool isolates the mathematical integration drift along the Z-axis (TVD) 
    by directly comparing the computed spatial trajectory against the validated operational 
    Excel benchmarks. It automatically generates a distinct CSV file for each processed well.

    Args:
        file_paths (list of str): Iterable of file paths pointing to target datasets.
        pipeline_runner (callable): Function reference to the execution pipeline to avoid circular imports.
        output_suffix (str, optional): Suffix appended to the dynamically generated filename. 
                                       Defaults to "_tvd_investigation.csv".

    Returns:
        pd.DataFrame or None: A concatenated Pandas DataFrame containing all extracted 
                              wellbore comparison data, or None if the batch completely fails.
    """
    all_dfs = []
    
    for path in file_paths:
        print(f"Processing TVD extraction for: {path}")
        
        # Dispatch the injected orchestration pipeline to resolve the spatial trajectory
        data = pipeline_runner(path) 
        
        if not data:
            print(f"  -> Skipping {path}: Pipeline extraction failed.")
            continue
            
        wellpath_data, df_out, term_covs, excel_covariances = data
        
        # Construct a localized spatial dataframe binding computed variables to benchmarks
        well_df = pd.DataFrame({
            'MD': wellpath_data['MD'],
            'Inclination': wellpath_data['Incl'],
            'Azimuth': wellpath_data['Azi'],
            'Calc_TVD': df_out['TVD'],
            'Excel_TVD': wellpath_data['TVD']
        })
        
        # Compute the absolute kinematic vertical displacement drift (Delta TVD)
        well_df['Delta_TVD'] = well_df['Calc_TVD'] - well_df['Excel_TVD']
        all_dfs.append(well_df)
        
        # Dynamically construct the unique filename for the current well and export
        base_name = os.path.splitext(os.path.basename(path))[0]
        filename = f"{base_name}{output_suffix}"
        well_df.to_csv(filename, index=False)
        print(f"  -> Successfully exported TVD data to: {filename}")
        
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n Successfully processed TVD investigation data for {len(all_dfs)} well(s).")
        return final_df
    else:
        print("\n No data was extracted from the provided file paths.")
        return None

def export_covariance_interval(df_out, term_covs, error_term, md_start, md_end, well_id="Well", export_csv=False, print_matrices=True):
    """
    Extracts specific covariance components for an error term over a given MD interval 
    into a Pandas DataFrame and prints matrices to the terminal.
    
    Args:
        df_out (pd.DataFrame): Trajectory dataframe containing Measured Depths.
        term_covs (dict): Dictionary of calculated covariance arrays.
        error_term (str): Target ISCWSA term (e.g., 'DECR').
        md_start (float): Minimum measured depth interval to extract.
        md_end (float): Maximum measured depth interval to extract.
        well_id (str): Identifier for the well to ensure unique file naming. Defaults to "Well".
        export_csv (bool): If True, saves the Pandas DataFrame to a CSV file. Defaults to False.
        print_matrices (bool): If True, prints the full 3x3 covariance matrix for each station. Defaults to True.

    Returns:
        pd.DataFrame: A formatted dataframe containing the selected matrix components.
    """
    if error_term not in term_covs:
        print(f"Error term '{error_term}' not found in calculated covariances.")
        return None
    
    # 1. Filter the DataFrame for the target depth interval
    mask = (df_out['MD'] >= md_start) & (df_out['MD'] <= md_end)
    mds = df_out.loc[mask, 'MD'].values
    
    # 2. Extract the subset of 3x3 matrices for the filtered interval
    matrices = term_covs[error_term][mask]
    
    # 3. Iterate through the interval to build the records and print matrices
    records = []
    for i, md in enumerate(mds):
        matrix = matrices[i]
        
        # Directly print the full 3x3 matrix to the terminal if requested
        if print_matrices:
            print(f"\n--- {error_term} Covariance Matrix at MD: {md:.2f}m ---")
            with np.printoptions(precision=6, suppress=True, formatter={'float': '{: 12.6e}'.format}):
                print(matrix)

        # Build the dictionary row for the Pandas DataFrame
        records.append({
            'MD': md,
            'Cov_NE': matrix[0, 1],
            'Cov_NV': matrix[0, 2],
            'Cov_EV': matrix[1, 2]
        })
        
    result_df = pd.DataFrame(records)
    if result_df.empty: 
        print(f"No data found for {error_term} between {md_start}m and {md_end}m.")
        return result_df
        
    # 4. Print the final summary table of the selected components
    print(f"\n--- Covariance Summary Table for {error_term} ({md_start}m to {md_end}m) ---")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    
    # 5. Export to CSV for manual validation if requested
    if export_csv:
        # Clean the file extension off the well_id to create a clean filename
        base_name = os.path.splitext(os.path.basename(well_id))[0]
        filename = f"{base_name}_{error_term}_validation.csv"
        
        result_df.to_csv(filename, index=False)
        print(f"\n-> Exported DataFrame to {filename}")
        
    return result_df
