import copy
import numpy as np
import pandas as pd

# =====================================================================
# CORE DATA EXTRACTION
# =====================================================================

def _load_model_sheet(file_path):
    """
    Helper function to load and clean the 'Model' sheet from the Excel workbook.
    Centralizes I/O logic to prevent reading the same sheet twice.

    Args:
        file_path (str): Path to the target Excel file.

    Returns:
        pd.DataFrame: Cleaned dataframe containing 'Code' and 'Magnitude' columns.
    """
    # Load the sheet skipping the first 2 header rows to reach the data
    df_model_raw = pd.read_excel(file_path, sheet_name='Model', header=2)
    
    # Strip whitespace and line breaks from column names to prevent KeyError
    df_model_raw.columns = df_model_raw.columns.astype(str).str.strip().str.replace('\n', '')
    
    # Return only the necessary columns
    return df_model_raw[['Code', 'Magnitude']].copy()

def extract_well_and_model_data(file_path):
    """
    Reads and cleans 'Wellpath' and 'Model' sheets from the input workbook.

    Args:
        file_path (str): Path to the Excel file containing well survey and model sheets.

    Returns:
        tuple: A pair of pandas DataFrames (df_wellpath, df_model), or (None, None) if an error occurs.
    """
    try:
        # Step 1: Read raw wellpath survey data, skipping the top row
        df_well_raw = pd.read_excel(file_path, sheet_name='Wellpath', header=1)
        df_well_raw.columns = df_well_raw.columns.astype(str).str.strip().str.replace('\n', '')
        
        # Step 2: Extract core spatial columns and rename them to standard conventions
        wellpath_columns = ['MD', 'Inc (°)', 'Az (°)', 'TVD']
        df_wellpath = df_well_raw[wellpath_columns].copy()
        df_wellpath = df_wellpath.rename(columns={'Inc (°)': 'Incl', 'Az (°)': 'Azi'})
        
        # Step 3: Load model parameters using the shared helper function
        df_model = _load_model_sheet(file_path)

        return df_wellpath, df_model
        
    except KeyError as e:
        print(f"\n[!] ERROR: Required columns not found. Found: {df_well_raw.columns.tolist()}")
        return None, None
    except Exception as e:
        print(f"Unexpected error extracting data: {e}")
        return None, None

# =====================================================================
# ERROR MODEL CONSTRUCTION
# =====================================================================

def build_custom_error_dict(model_data, base_error_dict):
    """
    Builds a custom error dictionary mapped directly to the extracted Model dataframe.

    Args:
        model_data (pd.DataFrame): Cleaned dataframe containing 'Code' and 'Magnitude' columns.
        base_error_dict (dict): Default dictionary defining base ISCWSA error structures.

    Returns:
        dict: Customized dictionary containing active error terms with updated magnitudes.
    """
    custom_dict = {}
    
    # Guard against empty extraction
    if model_data is None or model_data.empty:
        print("[!] Warning: Provided model data is empty.")
        return custom_dict

    # Step 1: Drop any rows missing code or magnitude values
    df_model = model_data.dropna(subset=['Code', 'Magnitude'])
    
    # Step 2: Iterate directly through model specifications to populate active dictionary
    for _, row in df_model.iterrows():
        term = str(row['Code']).strip()
        magnitude = float(row['Magnitude'])
        
        # Step 3: Match Excel error codes directly against the base error dictionary keys
        if term in base_error_dict:
            # Deepcopy to prevent mutating the original global dictionary
            custom_dict[term] = copy.deepcopy(base_error_dict[term])
            custom_dict[term]['mag'] = magnitude
            
    return custom_dict

# =====================================================================
# COVARIANCE BENCHMARKING
# =====================================================================

def _align_covariance_rows(df_cov, expected_mds, term, md_tolerance=1e-8):
    """Keep the covariance rows that correspond to physical Wellpath stations.

    The ISCWSA diagnostic sheets contain formulas below the populated Wellpath
    range. Excel caches those formulas as zero-MD rows, so ``dropna`` alone does
    not remove them. The physical Wellpath MD sequence is the authoritative row
    index for a station-by-station benchmark.
    """
    expected = np.asarray(expected_mds, dtype=float)
    observed = df_cov['MD'].to_numpy(dtype=float)

    if expected.ndim != 1 or len(expected) == 0 or not np.isfinite(expected).all():
        raise ValueError("Expected Wellpath MD values must be a finite non-empty vector.")
    if len(expected) > 1 and np.any(np.diff(expected) <= 0.0):
        raise ValueError("Expected Wellpath MD values must be strictly increasing.")
    if len(observed) < len(expected):
        raise ValueError(
            f"Reference sheet '{term}' has {len(observed)} covariance rows but "
            f"the Wellpath contains {len(expected)} stations."
        )

    physical_md = observed[:len(expected)]
    aligned = np.isclose(physical_md, expected, rtol=0.0, atol=md_tolerance)
    if not aligned.all():
        first_bad = int(np.flatnonzero(~aligned)[0])
        raise ValueError(
            f"Reference sheet '{term}' is not aligned with Wellpath at row "
            f"{first_bad}: covariance MD={physical_md[first_bad]:.12g}, "
            f"Wellpath MD={expected[first_bad]:.12g}."
        )

    result = df_cov.iloc[:len(expected)].copy().reset_index(drop=True)
    result.attrs['ignored_trailing_formula_rows'] = len(observed) - len(expected)
    return result


def extract_excel_covariances(file_path, active_terms, expected_mds=None):
    """
    Extracts reference NN, EE, VV, NE, NV, EV covariance matrices from designated Excel sheets.

    Args:
        file_path (str): Path to the Excel workbook.
        active_terms (list): List of active error term string identifiers to extract.
        expected_mds (array-like, optional): Physical MD sequence from Wellpath.
            When supplied, trailing diagnostic formulas outside that sequence
            are excluded only after the aligned prefix has been verified.

    Returns:
        dict: A dictionary mapping error term names to their respective reference covariance DataFrames.
    """
    # Load the workbook once to index available sheet names.
    excel_file = pd.ExcelFile(file_path)
    available_sheets = excel_file.sheet_names

    extracted_covs = {}
    expected_cols = ['NN', 'EE', 'VV', 'NE', 'NV', 'EV']
    
    # Step 1: Iterate through active error terms to locate and parse matching Excel sheets
    for term in active_terms:
        if term not in available_sheets:
            raise ValueError(f"Reference workbook is missing sheet '{term}'.")
        try:
            # The diagnostic sheets use a two-row header.
            df_headers = pd.read_excel(
                file_path,
                sheet_name=term,
                header=None,
                nrows=2,
            )
            top_row = df_headers.iloc[0].ffill().astype(str).str.strip()
            sub_row = df_headers.iloc[1].astype(str).str.strip()

            target_indices, col_names = [], []
            for column in range(len(top_row)):
                if top_row[column] == 'Covariance' and sub_row[column] in expected_cols:
                    target_indices.append(column)
                    col_names.append(sub_row[column])

            md_index = None
            for column in range(len(top_row)):
                top_name = top_row[column].strip().lower()
                sub_name = sub_row[column].strip().lower()
                if (
                    sub_name in {'md', 'measured depth'}
                    or top_name in {'md', 'measured depth'}
                ):
                    md_index = column
                    break

            if len(target_indices) != 6 or md_index is None:
                raise ValueError(
                    f"Sheet '{term}' is missing MD or covariance columns. "
                    f"Covariance columns found: {col_names}"
                )

            df_data = pd.read_excel(
                file_path,
                sheet_name=term,
                header=None,
                skiprows=2,
            )
            df_cov = df_data.iloc[:, [md_index] + target_indices].copy()
            df_cov.columns = ['MD'] + col_names
            df_cov = df_cov.dropna(subset=['MD']).reset_index(drop=True)
            df_cov = df_cov.apply(pd.to_numeric, errors='raise')

            # The first covariance row is mathematically zero in these files,
            # although some cells are stored as blanks.
            df_cov[expected_cols] = df_cov[expected_cols].fillna(0.0)
            if expected_mds is not None:
                df_cov = _align_covariance_rows(df_cov, expected_mds, term)
            extracted_covs[term] = df_cov
        except Exception as exc:
            raise ValueError(
                f"Failed to extract covariance reference sheet '{term}'."
            ) from exc
                
    return extracted_covs

# =====================================================================
# MAIN EXECUTION WRAPPER
# =====================================================================

def run_extraction(file_path, base_error_dict):
    """
    A unified wrapper function. Extracts trajectory, compiles the error model, extracts benchmarks 
    and handles unit conversions.

    Args:
        file_path (str): Path to the Excel file.
        base_error_dict (dict): The default standard error model dictionary.

    Returns:
        tuple: (wellpath_data DataFrame, custom error model dictionary, benchmark covariance dictionary).
               Returns (None, None, None) on failure.
    """
    # Step 1: Read raw data sources from the workbook
    wellpath_data, model_data = extract_well_and_model_data(file_path)
    if wellpath_data is None: 
        return None, None, None

    # Keep the workbook-unit MD sequence for aligning diagnostic sheets before
    # the optional feet-to-metres conversion.
    reference_mds = wellpath_data['MD'].to_numpy(dtype=float).copy()

    # Step 2: Handle metric conversions if the ISCWSA-2 (feet) dataset is detected.
    is_feet = "iscwsa-2" in file_path.lower()
    if is_feet:
        wellpath_data['MD'] *= 0.3048
        if 'TVD' in wellpath_data.columns:
            wellpath_data['TVD'] *= 0.3048

    # Step 3: Generate the operational error dictionary based on the Model sheet
    custom_dict = build_custom_error_dict(model_data, base_error_dict)
    
    # Step 4: Extract reference benchmark sheets for validation
    excel_covariances = extract_excel_covariances(
        file_path,
        list(custom_dict.keys()),
        expected_mds=reference_mds,
    )
    if is_feet:
        for df_cov in excel_covariances.values():
            df_cov['MD'] *= 0.3048

    return wellpath_data, custom_dict, excel_covariances
