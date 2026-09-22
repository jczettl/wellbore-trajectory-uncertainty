import numpy as np
import pandas as pd
from core_math import get_jacobian

# =====================================================================
# JACOBIAN EXTRACTION & VECTOR MAPPING
# =====================================================================

def _get_jacobian_vectors(jacobian_step, axis):
    """
    Extracts the correct partial derivative vectors from the 3x5 Jacobian matrix 
    based on the specific measurement axis being evaluated.

    Args:
        jacobian_step (np.ndarray): The 3x5 Jacobian matrix for the current segment.
        axis (str): The primary measurement axis causing the error ('Inc', 'Azi', or 'MD').

    Returns:
        tuple: A pair of arrays representing the (j_prev, j_curr) vectors. 
               For 'MD' (Measured Depth), j_prev is returned as None since length 
               is a scalar step, not a differential node.
    """
    if axis == 'MD':
        # Depth errors exclusively use the 5th column (index 4) for the step length derivative
        return None, jacobian_step[:, 4]
    elif axis == 'Inc':
        # Inclination uses col 0 for the previous station and col 2 for the current station
        return jacobian_step[:, 0], jacobian_step[:, 2]
    elif axis == 'Azi':
        # Azimuth uses col 1 for the previous station and col 3 for the current station
        return jacobian_step[:, 1], jacobian_step[:, 3]
    raise ValueError("axis must be 'MD', 'Inc', or 'Azi'.")


# =====================================================================
# FUNCTIONAL ERROR PROPAGATION ENGINE
# =====================================================================

def calculate_systematic_error(current_state, jacobian_step, sigma, w_prev, w_curr, axis):
    """
    Calculate a correlated covariance term (modes S, W, and G).
    
    For these correlated modes, physical error vectors are accumulated before
    their outer product is formed.

    Args:
        current_state (dict): Immutable state dictionary containing 'accum' and 'pending' vectors.
        jacobian_step (np.ndarray): The 3x5 Jacobian matrix for the segment.
        sigma (float): The base statistical magnitude of the error term (usually derived from standard deviation).
        w_prev (float): Weighting function output evaluated for the previous station.
        w_curr (float): Weighting function output evaluated for the current station.
        axis (str): The physical measurement axis being perturbed ('Inc', 'Azi', 'MD').

    Returns:
        tuple: Updated state and the 3x3 covariance at the current station. The
               state includes ``e_total``, the complete vector represented by
               that covariance.
    """
    # Clone state variables
    e_accum = current_state['e_accum'].copy()
    e_prev_leg = current_state['e_prev_leg'].copy()
    
    J_prev, J_curr = _get_jacobian_vectors(jacobian_step, axis)
    
    # Calculate the base error vector strictly at the current node (e_k)
    e_curr_leg = sigma * w_curr * J_curr

    # 1. Correlated Angular Errors -> Direct Vector Addition
    if axis != 'MD':
        e_prev_tot = e_prev_leg + (sigma * w_prev * J_prev)
        e_accum += e_prev_tot # Summing raw vectors to maintain phase correlation
        e_prev_leg = e_curr_leg.copy()
        
        temp_accum = e_accum + e_curr_leg
        e_total = temp_accum
        C_tot = np.outer(e_total, e_total)
        
    # 2. Single-Node Depth Errors -> Localized Accumulation
    else:
        e_accum += e_curr_leg # Summing vectors
        e_total = e_accum.copy()
        C_tot = np.outer(e_total, e_total)

    return {
        'e_accum': e_accum,
        'e_prev_leg': e_prev_leg,
        'e_curr_leg': e_curr_leg,
        'e_total': e_total,
    }, C_tot

def calculate_random_error(current_state, jacobian_step, sigma, w_prev, w_curr, axis, is_drfr=False):
    """
    Calculates random covariance terms (mode R).
    
    Random errors are completely uncorrelated between individual surveys. Therefore, 
    the localized covariance matrices are calculated first at each node, and then the 
    matrices themselves are summed (sum of squares) to accumulate the uncertainty.

    Args:
        current_state (dict): State dictionary containing 'C' (covariance matrix), 
                              'pending' vectors, and 'is_first_anchor' flag.
        jacobian_step (np.ndarray): The 3x5 Jacobian matrix.
        sigma (float): Base error magnitude scalar.
        w_prev (float): Weight evaluated for the previous station.
        w_curr (float): Weight evaluated for the current station.
        axis (str): The physical measurement axis.
        is_drfr (bool): Special flag explicitly targeting the Depth Reference Error (DRFR) logic.

    Returns:
        tuple: A pair containing the updated state dictionary, and the resulting 
               3x3 Covariance Matrix for the current step.
    """
    # Clone state variables
    C_rand = current_state['C_rand'].copy()
    e_prev_leg = current_state['e_prev_leg'].copy()
    is_first = current_state['is_first_anchor']
    
    J_prev, J_curr = _get_jacobian_vectors(jacobian_step, axis)
    
    # Calculate the base error vector strictly at the current node (e_k)
    e_curr_leg = sigma * w_curr * J_curr
    
    # 1. Depth Reference Error (DRFR) -> Matrix Subtraction Mode
    if is_drfr:
        if is_first and np.all(e_prev_leg == 0):
            e_prev_leg = e_curr_leg.copy()
            C_tot = C_rand + np.outer(e_curr_leg, e_curr_leg)
        else:
            e_prev_tot = e_prev_leg - e_curr_leg
            C_rand += np.outer(e_prev_tot, e_prev_tot) # Summing matrices directly
            e_prev_leg = e_curr_leg.copy()
            C_tot = C_rand + np.outer(e_curr_leg, e_curr_leg)
            
    # 2. Uncorrelated Angular Errors -> Matrix Addition Mode
    elif axis != 'MD':
        e_prev_tot = e_prev_leg + (sigma * w_prev * J_prev)
        C_rand += np.outer(e_prev_tot, e_prev_tot) # Summing matrices directly
        e_prev_leg = e_curr_leg.copy()
        C_tot = C_rand + np.outer(e_curr_leg, e_curr_leg)
        
    # 3. Single-Node Depth Errors -> Localized Matrix Accumulation
    else:
        C_rand += np.outer(e_curr_leg, e_curr_leg) # Summing matrices directly
        C_tot = C_rand.copy()

    return {
        'C_rand': C_rand,
        'e_prev_leg': e_prev_leg,
        'e_curr_leg': e_curr_leg,
        'is_first_anchor': False,
    }, C_tot


# =====================================================================
# SPATIAL INTERPOLATION
# =====================================================================

def _fill_array_gaps(data_array, df_out):
    """
    A generic linear interpolation wrapper capable of filling data gaps across both 
    1D vectors and multi-dimensional matrices situated between actual survey anchor stations.
    
    This function ensures that artificially injected grid points (used for high-resolution 
    3D visualization) possess valid, smoothly transitioning data profiles rather 
    than stepping abruptly.

    Args:
        data_array (np.ndarray): An N-dimensional array containing calculated 
                                 values strictly at anchor stations.
        df_out (pd.DataFrame): The trajectory dataframe containing an 'is_interpolated' 
                               boolean column to identify structural grid points.

    Returns:
        np.ndarray: The modified array with fully populated values across all grid points.
    """
    # Locate exact indices of actual (non-interpolated) physical surveys
    anchor_indices = df_out.index[~df_out['is_interpolated']].tolist()
    
    for i in range(len(anchor_indices) - 1):
        idx_start, idx_end = anchor_indices[i], anchor_indices[i+1]
        val_start, val_end = data_array[idx_start], data_array[idx_end]
        
        steps = idx_end - idx_start
        for step in range(1, steps):
            f = step / steps
            # Execute Linear Interpolation (LERP) across the entire data structure
            data_array[idx_start + step] = val_start * (1 - f) + val_end * f
            
    return data_array


# =====================================================================
# ISCWSA COVARIANCE ORCHESTRATION
# =====================================================================

# Functional dispatch dictionary routing error modes to their appropriate calculators
ERROR_CALCULATORS = {
    'S': calculate_systematic_error,
    'W': calculate_systematic_error,
    'G': calculate_systematic_error,
    'R': calculate_random_error
}

def calculate_iscwsa_covariance(df, error_dict, md_col='MD', inc_col='Incl', azi_col='Azi',balanced_tangential_jacobi_approximation=False):
    """
    Iterates sequentially through anchor stations, executing the ISCWSA error model.
    
    Builds the cumulative 3x3 position covariance and retains term-level vectors
    used by the diagnostic plots.

    Args:
        df (pd.DataFrame): Input wellbore trajectory dataframe.
        error_dict (dict): A dictionary defining the active ISCWSA error terms to evaluate. 
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        inc_col (str): Column name for Inclination. Defaults to 'Incl'.
        azi_col (str): Column name for Azimuth. Defaults to 'Azi'.

    Returns:
        tuple: A triple containing:
               - C_total_array (np.ndarray): The accumulated (N, 3, 3) covariance matrix array for all combined terms.
               - term_covariances (dict): A dictionary mapping each individual error term to its isolated (N, 3, 3) matrix array.
               - term_diagnostics (dict): Arrays including ``e_accum``, complete
                 ``e_total`` vectors, current ``e_step`` vectors, and ``C_rand``.
    """
    if not error_dict:
        raise ValueError("error_dict must contain at least one error term.")

    required_columns = [md_col, inc_col, azi_col]
    missing_columns = [name for name in required_columns if name not in df.columns]
    if missing_columns:
        raise ValueError(
            "Trajectory is missing required covariance columns: "
            + ", ".join(missing_columns)
        )

    df_out = df.copy()
    if 'is_interpolated' not in df_out.columns:
        df_out['is_interpolated'] = False
        
    n_stations = len(df_out)
    mds = df_out[md_col].to_numpy(dtype=float)
    incs = df_out[inc_col].to_numpy(dtype=float)
    azis = df_out[azi_col].to_numpy(dtype=float)
    if n_stations == 0 or not np.isfinite(np.column_stack([mds, incs, azis])).all():
        raise ValueError("Trajectory covariance input is empty or contains non-finite values.")
    
    # Pre-allocate zero arrays for high-performance memory management
    term_covariances = {term: np.zeros((n_stations, 3, 3)) for term in error_dict}
    C_total_array = np.zeros((n_stations, 3, 3))
    
    term_diagnostics = {
        term: {
            'e_accum': np.zeros((n_stations, 3)),
            'e_total': np.zeros((n_stations, 3)),
            'e_step': np.zeros((n_stations, 3)),
            'C_rand': np.zeros((n_stations, 3, 3))
        } for term in error_dict
    }
    
    # Validate the model once so configuration errors do not depend on the
    # number of trajectory stations.
    for term, props in error_dict.items():
        missing_properties = {'mag', 'axis', 'mode', 'weight'}.difference(props)
        if missing_properties:
            raise ValueError(
                f"Error term '{term}' is missing properties: "
                + ", ".join(sorted(missing_properties))
            )
        if props['axis'] not in {'MD', 'Inc', 'Azi'}:
            raise ValueError(f"Unsupported measurement axis for term '{term}'.")
        if props['mode'] not in ERROR_CALCULATORS:
            raise ValueError(f"Unsupported propagation mode for term '{term}'.")
        if not np.isfinite(props['mag']) or props['mag'] < 0.0:
            raise ValueError(f"Error magnitude for term '{term}' must be finite and non-negative.")
        if not callable(props['weight']):
            raise ValueError(f"Weight for term '{term}' must be callable.")

    # Initialize the state tracking dictionaries by propagation mode.
    states = {}
    for term, props in error_dict.items():
        if props['mode'] in ['S', 'W', 'G']:
            states[term] = {
                'e_accum': np.zeros(3), 
                'e_prev_leg': np.zeros(3),
                'is_first_anchor': True
            }
        elif props['mode'] == 'R':
            states[term] = {
                'C_rand': np.zeros((3, 3)), 
                'e_prev_leg': np.zeros(3), 
                'is_first_anchor': True
            }

    anchor_indices = df_out.index[~df_out['is_interpolated']].tolist()
    if not anchor_indices:
        raise ValueError("Trajectory contains no physical survey stations.")
    if len(anchor_indices) > 1 and np.any(np.diff(mds[anchor_indices]) <= 0.0):
        raise ValueError("Physical survey-station MD values must be strictly increasing.")
    
    # Step sequentially through every physical survey anchor station
    for i in range(1, len(anchor_indices)):
        idx_prev, idx_curr = anchor_indices[i-1], anchor_indices[i]
        dl_anchor = mds[idx_curr] - mds[idx_prev]
        
        J_step = get_jacobian(incs[idx_prev], incs[idx_curr], azis[idx_prev], azis[idx_curr], dl_anchor, balanced_tangential_jacobi_approximation=balanced_tangential_jacobi_approximation)
        row_prev, row_curr = df_out.iloc[idx_prev], df_out.iloc[idx_curr]

        for term, props in error_dict.items():
            mode, axis, mag = props['mode'], props['axis'], props['mag']
            
            w_prev = props['weight'](row_prev, row_prev, dl_anchor) if axis != 'MD' else None
            w_curr = props['weight'](row_curr, row_prev, dl_anchor)
            weights = [w_curr] if w_prev is None else [w_prev, w_curr]
            if not np.isfinite(weights).all():
                raise ValueError(f"Non-finite weight generated for term '{term}'.")
            
            calc_func = ERROR_CALCULATORS.get(mode)
            if calc_func is None:
                raise ValueError(
                    f"Unsupported propagation mode '{mode}' for term '{term}'. "
                    "Expected R, S, W, or G."
                )

            # Functional dispatch strictly separates mathematical computation from loop logic
            if mode in ['S', 'W', 'G']:
                new_state, current_cov = calc_func(states[term], J_step, mag, w_prev, w_curr, axis)
            elif mode == 'R':
                new_state, current_cov = calc_func(states[term], J_step, mag, w_prev, w_curr, axis, is_drfr=(term == 'DRFR'))

            # Record updated states and map covariances
            states[term] = new_state
            term_covariances[term][idx_curr] = current_cov
            C_total_array[idx_curr] += current_cov
            
            # Map raw 1D diagnostic vectors for structural analysis
            term_diagnostics[term]['e_accum'][idx_curr] = new_state.get('e_accum', np.zeros(3))
            term_diagnostics[term]['e_total'][idx_curr] = new_state.get('e_total', np.zeros(3))
            term_diagnostics[term]['e_step'][idx_curr] = new_state.get('e_curr_leg', np.zeros(3))
            term_diagnostics[term]['C_rand'][idx_curr] = new_state.get('C_rand', np.zeros((3, 3)))

    # Backfill missing structural grid points using the LERP utility across all tracking arrays
    for term in error_dict:
        term_covariances[term] = _fill_array_gaps(term_covariances[term], df_out)
        term_diagnostics[term]['e_accum'] = _fill_array_gaps(term_diagnostics[term]['e_accum'], df_out)
        term_diagnostics[term]['e_total'] = _fill_array_gaps(term_diagnostics[term]['e_total'], df_out)
        term_diagnostics[term]['e_step'] = _fill_array_gaps(term_diagnostics[term]['e_step'], df_out)
        term_diagnostics[term]['C_rand'] = _fill_array_gaps(term_diagnostics[term]['C_rand'], df_out)

    C_total_array = _fill_array_gaps(C_total_array, df_out)
        
    return C_total_array, term_covariances, term_diagnostics


# =====================================================================
# UTILITIES & DIAGNOSTICS
# =====================================================================

def get_term_covariance_at_md(df, term_covariances, target_md, error_term):
    """
    Utility lookup function to retrieve and format the precise 3x3 Covariance Matrix 
    for a chosen ISCWSA error term at a specified Measured Depth.

    Args:
        df (pd.DataFrame): The main wellbore trajectory dataframe.
        term_covariances (dict): The dictionary of individual term covariance arrays 
                                 returned by the `calculate_iscwsa_covariance` function.
        target_md (float): The measured depth (MD) target to evaluate.
        error_term (str): The string identifier of the target ISCWSA term (e.g., 'DRFR').

    Returns:
        tuple: A pair containing:
               - actual_md (float): The exact MD of the station matched from the dataframe.
               - cov_matrix (np.ndarray): The 3x3 Covariance Matrix explicitly mapped to that station.
    """
    if error_term not in term_covariances:
        valid_terms = ", ".join(term_covariances.keys())
        raise KeyError(f"Error term '{error_term}' not found. Available terms: {valid_terms}")
        
    # Snap search dynamically to the closest recorded dataframe index based on provided depth
    idx = (df['MD'] - target_md).abs().idxmin()
    actual_md = df.loc[idx, 'MD']
    
    # Introduce a safe float comparison boundary to warn of imprecise snapping
    if not np.isclose(actual_md, target_md, atol=0.01):
        print(f"\n[!] Warning: Exact MD {target_md}m not found. Snapping to nearest station at {actual_md:.2f}m.")
        
    cov_matrix = term_covariances[error_term][idx]
    
    # Clean terminal formatting for immediate engineering validation
    print(f"\n" + "-"*50)
    print(f" ISCWSA Verification: [{error_term}] at MD {actual_md:.2f}m")
    print("-"*50)
    print("Covariance Matrix (NEV in square meters):")
    
    with np.printoptions(precision=6, suppress=True, formatter={'float': '{: 10.6f}'.format}):
        print(cov_matrix)
    print("-"*50)
    
    return actual_md, cov_matrix
