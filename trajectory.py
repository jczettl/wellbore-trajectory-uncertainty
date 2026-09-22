import numpy as np
import pandas as pd

from core_math import (
    min_curvature_step, 
    slerp_interpolate_vectors, 
    get_dogleg_angle
)


# =====================================================================
# TRAJECTORY INTERPOLATION
# =====================================================================

def interpolate_trajectory_grid(df, step_size=None, num_points=None, md_col='MD', inc_col='Incl', azi_col='Azi'):
    """
    Generates a high-resolution uniform structural grid across sparse survey rows.
    
    This function inserts interpolated stations between existing survey points 
    using Spherical Linear Interpolation (SLERP) to maintain the circular-arc 
    assumption required by the Minimum Curvature method.

    Args:
        df (pd.DataFrame): Input wellbore trajectory dataframe.
        step_size (float, optional): Desired measured depth (MD) interval between points.
        num_points (int, optional): Exact number of points to insert between each station pair.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        inc_col (str): Column name for Inclination. Defaults to 'Incl'.
        azi_col (str): Column name for Azimuth. Defaults to 'Azi'.

    Returns:
        pd.DataFrame: A unified dataframe of original and interpolated stations, 
                      ordered by MD and containing an 'is_interpolated' flag.
    """
    # Enforce mutually exclusive arguments to prevent conflicting interpolation logic
    if step_size is not None and num_points is not None:
        raise ValueError("Specify either 'step_size' or 'num_points', not both.")
        
    interpolated_rows = []
    
    # DataFrame iteration is acceptable here strictly for data assembly
    for k in range(len(df) - 1):
        row1, row2 = df.iloc[k], df.iloc[k+1]
        md1, md2 = row1[md_col], row2[md_col]
        
        # Extract the starting row as a dictionary and flag it as an original data point
        row1_dict = row1.to_dict()
        row1_dict['is_interpolated'] = False
        interpolated_rows.append(row1_dict)
        
        # Calculate segment length
        delta_md = md2 - md1
        
        # Skip interpolation if the stations are effectively at the exact same depth
        if np.isclose(delta_md, 0.0, atol=1e-3):
            continue
            
        # Determine the target Measured Depths for interpolation based on the chosen method
        if step_size is not None:
            num_segments = max(1, int(round(delta_md / step_size)))
            target_mds = np.linspace(md1, md2, num_segments + 1)[1:-1]
        elif num_points is not None:
            target_mds = np.linspace(md1, md2, num_points + 2)[1:-1]
        else:
            target_mds = []
            
        # Perform SLERP for each target MD within the current segment
        for m_int in target_mds:
            # Calculate the fractional distance (f) along the current wellbore curve segment
            f = (m_int - md1) / delta_md
            
            # Interpolate inclination and azimuth to maintain constant angular velocity along the arc
            i_int, a_int = slerp_interpolate_vectors(row1[inc_col], row1[azi_col], row2[inc_col], row2[azi_col], f)
            
            # Construct the newly interpolated row
            new_row = row1.copy()
            new_row[md_col], new_row[inc_col], new_row[azi_col] = m_int, i_int, a_int
            new_row_dict = new_row.to_dict()
            new_row_dict['is_interpolated'] = True  
            interpolated_rows.append(new_row_dict)
            
    # Append the final original station to cap off the trajectory
    last_row_dict = df.iloc[-1].to_dict()
    last_row_dict['is_interpolated'] = False
    interpolated_rows.append(last_row_dict)
    
    # Deduplicate in case of overlapping MDs and reset the index for a clean output
    return pd.DataFrame(interpolated_rows).drop_duplicates(subset=[md_col]).reset_index(drop=True)


# =====================================================================
# SPATIAL TRAJECTORY CALCULATION
# =====================================================================

def calculate_spatial_trajectory(df, md_col='MD', inc_col='Incl', azi_col='Azi', 
                                 ns_col='NS', ew_col='EW', tvd_col='TVD', 
                                 start_coords=(0.0, 0.0, 0.0), dls_interval=30.0):
    """
    Computes physical spatial positioning (North, East, True Vertical Depth) 
    and Dogleg Severity (DLS) purely over NumPy arrays for performance.
    
    This function steps through the wellbore path, computing the Minimum Curvature 
    displacement for each interval and accumulating it to find global coordinates.

    Args:
        df (pd.DataFrame): The input dataframe containing the survey stations.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        inc_col (str): Column name for Inclination (degrees). Defaults to 'Incl'.
        azi_col (str): Column name for Azimuth (degrees). Defaults to 'Azi'.
        ns_col (str): Target column name for North-South output. Defaults to 'NS'.
        ew_col (str): Target column name for East-West output. Defaults to 'EW'.
        tvd_col (str): Target column name for True Vertical Depth output. Defaults to 'TVD'.
        start_coords (tuple): Starting coordinate tie-in (North, East, TVD). Defaults to (0.0, 0.0, 0.0).
        dls_interval (float): The standard interval length (e.g., 30m or 100ft) over which 
                              Dogleg Severity is normalized. Defaults to 30.0.

    Returns:
        pd.DataFrame: A copy of the input dataframe with newly calculated 
                      coordinate and DLS columns appended.
    """
    df_out = df.copy()
    n_stations = len(df_out)
    
    # Initialize output arrays with zeros for memory allocation and performance
    ns_coords, ew_coords, tvd_coords = np.zeros(n_stations), np.zeros(n_stations), np.zeros(n_stations)
    dls_coords = np.zeros(n_stations)  
    
    # Apply the starting tie-in coordinates to the first station
    ns_coords[0], ew_coords[0], tvd_coords[0] = start_coords
    
    # Extract columns into flat NumPy arrays for faster mathematical iteration
    mds = df_out[md_col].values
    incs = df_out[inc_col].values
    azis = df_out[azi_col].values
    
    for k in range(1, n_stations):
        # Calculate the course length (Delta MD) between the current and previous station
        dl = mds[k] - mds[k-1]
        
        # Calculate spatial displacement via Minimum Curvature Method
        delta_pos = min_curvature_step(incs[k-1], incs[k], azis[k-1], azis[k], dl) 
        
        # Accumulate the calculated deltas onto the previous station's global coordinates
        ns_coords[k] = ns_coords[k-1] + delta_pos[0]
        ew_coords[k] = ew_coords[k-1] + delta_pos[1]
        tvd_coords[k] = tvd_coords[k-1] + delta_pos[2]
        
        # Calculate DLS and store it (protecting against divide-by-zero on zero-length intervals)
        if dl > 0:
            # Convert degrees to radians for internal trigonometric calculations
            i1, i2 = np.radians(incs[k-1]), np.radians(incs[k])
            a1, a2 = np.radians(azis[k-1]), np.radians(azis[k])
            
            # Compute the dogleg angle (beta)
            beta = get_dogleg_angle(i1, i2, a1, a2)
            
            # Convert beta back to degrees and normalize it to the specified DLS interval length
            dls_coords[k] = np.degrees(beta) * (dls_interval / dl)
        
    # Bind the fully populated NumPy arrays back to the Pandas DataFrame
    df_out[ns_col] = ns_coords
    df_out[ew_col] = ew_coords
    df_out[tvd_col] = tvd_coords
    df_out['DLS'] = dls_coords 
    
    return df_out