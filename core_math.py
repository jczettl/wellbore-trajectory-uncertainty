import numpy as np

# =====================================================================
# PRIMITIVE VECTOR & ANGULAR MATHEMATICS
# =====================================================================

def _get_tangent_vector(i, a):
    """
    Converts spherical survey angles into a 3D Cartesian unit tangent vector.
    
    The resulting vector aligns with the North-East-Down (NED) coordinate system.

    Args:
        i (float): Inclination angle in radians.
        a (float): Azimuth angle in radians.

    Returns:
        np.ndarray: A 3-element array representing the unit tangent vector 
                    in the format [North, East, TVD].
    """
    return np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])

def get_dogleg_angle(i1, i2, a1, a2):
    """
    Calculates the dogleg angle (beta) between two survey stations.
    Using Harversine formula prevents catastrophic cancellation 
    in nearly straight hole sections where the angle change is 
    extremely small.

    Args:
        i1 (float): Inclination at survey 1 in radians.
        i2 (float): Inclination at survey 2 in radians.
        a1 (float): Azimuth at survey 1 in radians.
        a2 (float): Azimuth at survey 2 in radians.

    Returns:
        float: The total dogleg angle (beta) in radians.
    """
    sin_half_i_diff = np.sin((i2 - i1) / 2.0)
    sin_half_a_diff = np.sin((a2 - a1) / 2.0)
    sin_sq_half_beta = sin_half_i_diff**2 + np.sin(i1) * np.sin(i2) * sin_half_a_diff**2
    
    # Clip to [0, 1] to handle potential floating-point inaccuracies before taking the square root
    return 2.0 * np.arcsin(np.sqrt(np.clip(sin_sq_half_beta, 0.0, 1.0)))

def get_rotation_matrix(inc, azi):
    """
    Calculates the 3x3 transformation matrix to map local cross-sectional coordinates 
    (High-Side, Right-Side, Tangent) to the global 3D coordinate system (North, East, TVD).
    
    Coordinate System Note: 
    Directional drilling uses a North-East-Down (NED) convention where Z is 
    positive downwards (TVD). The columns of this matrix represent the 
    local unit vectors mapped to global space:
      - Col 1 (X-local): High-Side. Note the -sin(i) Z-component points it "up" (negative TVD).
      - Col 2 (Y-local): Right-Side. Lies entirely in the horizontal plane (Z=0).
      - Col 3 (Z-local): Tangent. Points down the wellbore path.

    Args:
        inc (float): Inclination angle (degrees).
        azi (float): Azimuth angle (degrees).

    Returns:
        np.ndarray: A 3x3 rotation matrix mapping Local -> Global.
    """
    # Convert inputs to radians for internal trigonometric calculations
    i, a = np.radians(inc), np.radians(azi)

    # Standard Euler rotation matrix R = Rz(azi) * Ry(inc)
    return np.array([
        [np.cos(i) * np.cos(a), -np.sin(a), np.sin(i) * np.cos(a)],
        [np.cos(i) * np.sin(a),  np.cos(a), np.sin(i) * np.sin(a)],
        [-np.sin(i),            0,          np.cos(i)]
    ])


# =====================================================================
# TRAJECTORY & SPATIAL INTERPOLATION
# =====================================================================

def min_curvature_step(inc1, inc2, azi1, azi2, dl):
    """
    Calculates the displacement step (North, East, TVD) using the Minimum Curvature Method.
    
    This method assumes the wellbore path between two survey stations is a 
    circular arc on a plane defined by the two station vectors.

    Args:
        inc1 (float): Inclination at survey 1 (degrees).
        inc2 (float): Inclination at survey 2 (degrees).
        azi1 (float): Azimuth at survey 1 (degrees).
        azi2 (float): Azimuth at survey 2 (degrees).
        dl (float): Measured depth increment (Course Length) between surveys.

    Returns:
        np.ndarray: A 3-element array [Delta North, Delta East, Delta True Vertical Depth].
    """
    # Convert degrees to radians for trigonometric functions
    i1, i2 = np.radians(inc1), np.radians(inc2)
    a1, a2 = np.radians(azi1), np.radians(azi2)
    
    beta = get_dogleg_angle(i1, i2, a1, a2)
    # Calculate the Ratio Factor (RF). This factor smooths the straight-line tangential 
    # calculation into a circular arc.
    if np.isclose(beta, 0.0, atol=1e-4):
        # Taylor series expansion for extremely small dogleg angles (nearly straight hole).
        # As beta approaches 0, tan(beta/2)/(beta/2) approaches 1, but calculating it directly 
        # causes a divide-by-zero or massive precision loss.
        RF = 1.0 + (beta**2) / 12.0 + (beta**4) / 120.0
    else:
        # Standard minimum curvature ratio factor
        RF = (2.0 / beta) * np.tan(beta / 2.0)
    
    # Calculate displacement increments. 
    # This is essentially the Average Angle method multiplied by the Ratio Factor (f).
    dn = (dl / 2.0) * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * RF
    de = (dl / 2.0) * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * RF
    dtvd = (dl / 2.0) * (np.cos(i1) + np.cos(i2)) * RF
    return np.array([dn, de, dtvd])

def slerp_interpolate_vectors(inc1, azi1, inc2, azi2, f):
    """
    Executes Spherical Linear Interpolation (SLERP) to find the inclination and 
    azimuth at a fractional point between two survey stations.

    Args:
        inc1 (float): Inclination at survey 1 (degrees).
        azi1 (float): Azimuth at survey 1 (degrees).
        inc2 (float): Inclination at survey 2 (degrees).
        azi2 (float): Azimuth at survey 2 (degrees).
        f (float): Fractional distance along the arc (0.0 = survey 1, 1.0 = survey 2).

    Returns:
        tuple: A pair of floats representing the interpolated (inclination, azimuth) in degrees.
    """
    # Convert inputs to radians for internal trigonometric calculations
    i1, i2 = np.radians(inc1), np.radians(inc2)
    a1, a2 = np.radians(azi1), np.radians(azi2)
    
    # Convert angular spherical coordinates to 3D Cartesian unit tangent vectors
    v1 = _get_tangent_vector(i1, a1)
    v2 = _get_tangent_vector(i2, a2)
    
    beta = get_dogleg_angle(i1, i2, a1, a2)
    
    # Apply interpolation
    if np.isclose(beta, 0.0, atol=1e-4):
        # Fallback to standard Linear Interpolation (LERP) for nearly straight hole sections
        # to avoid dividing by sin(beta) -> 0.
        t_int = (1 - f) * v1 + f * v2
    else:
        # Standard SLERP formula
        t_int = (np.sin((1 - f) * beta) / np.sin(beta)) * v1 + (np.sin(f * beta) / np.sin(beta)) * v2
        
    # Re-normalize the interpolated vector (to correct for any floating-point drift)
    t_int /= np.linalg.norm(t_int)

    # Convert the interpolated 3D Cartesian vector back to spherical degrees (Inc, Azi)
    inc_int = np.degrees(np.arccos(t_int[2]))
    azi_int = np.degrees(np.arctan2(t_int[1], t_int[0])) % 360.0
    return inc_int, azi_int


# =====================================================================
# CALCULUS & ERROR PROPAGATION 
# =====================================================================

def get_jacobian(inc1, inc2, azi1, azi2, dl, balanced_tangential_jacobi_approximation=False):
    """
    Computes the exact analytical 3x5 Jacobian matrix for a Minimum Curvature step 
    or average angle approximation if average_angle_jacobi_approximation=True.
    
    The Jacobian represents the partial derivatives of the displacement outputs 
    (Delta North, East, TVD) with respect to the 5 inputs (inc1, azi1, inc2, azi2, dl).

    Args:
        inc1 (float): Inclination at survey 1 (degrees).
        inc2 (float): Inclination at survey 2 (degrees).
        azi1 (float): Azimuth at survey 1 (degrees).
        azi2 (float): Azimuth at survey 2 (degrees).
        dl (float): Measured depth increment between surveys.
    Returns:
        np.ndarray: A 3x5 Jacobian matrix.
                    Rows: [dNorth, dEast, dTVD]
                    Cols: [d_inc1, d_azi1, d_inc2, d_azi2, d_dl]
    """
    # Convert inputs to radians for internal trigonometric calculations
    i1, i2 = np.radians(inc1), np.radians(inc2)
    a1, a2 = np.radians(azi1), np.radians(azi2)
    
    # Unit tangent vectors at station 1 and station 2
    v1 = _get_tangent_vector(i1, a1)
    v2 = _get_tangent_vector(i2, a2)
    
    # Partial derivatives of the unit vectors w.r.t inclination and azimuth
    dv1_di1 = np.array([np.cos(i1)*np.cos(a1), np.cos(i1)*np.sin(a1), -np.sin(i1)])
    dv1_da1 = np.array([-np.sin(i1)*np.sin(a1), np.sin(i1)*np.cos(a1), 0.0])
    
    dv2_di2 = np.array([np.cos(i2)*np.cos(a2), np.cos(i2)*np.sin(a2), -np.sin(i2)])
    dv2_da2 = np.array([-np.sin(i2)*np.sin(a2), np.sin(i2)*np.cos(a2), 0.0])
    
    beta = get_dogleg_angle(i1, i2, a1, a2)

    # Partial derivatives of cos(beta) w.r.t the input angles
    dcb_di1 = -np.sin(i1)*np.cos(i2) + np.cos(i1)*np.sin(i2)*np.cos(a2-a1)
    dcb_di2 = -np.cos(i1)*np.sin(i2) + np.sin(i1)*np.cos(i2)*np.cos(a2-a1)
    dcb_da1 = np.sin(i1)*np.sin(i2)*np.sin(a2-a1)
    dcb_da2 = -dcb_da1

    if balanced_tangential_jacobi_approximation:
        # Construct the final Jacobian columns using an average angle approximation:
        J_dl = 0.5 * ( v1 + v2 )
        J_i1 = ( dl / 2.0) * dv1_di1
        J_a1 = ( dl / 2.0) * dv1_da1
        J_i2 = ( dl / 2.0) * dv2_di2
        J_a2 = ( dl / 2.0) * dv2_da2

    else:
        # Calculate Ratio Factor (RF) and its derivative w.r.t cos(beta)
        if np.isclose(beta, 0.0, atol=1e-4):
            # Taylor series expansion for small doglegs to prevent divide-by-zero errors.
            # Limits are evaluated as beta approaches 0.
            RF = 1.0 + (beta**2) / 12.0 + (beta**4) / 120.0
            dRF_dcb = -1.0 / 6.0 - (beta**2)*11.0/180.0 
        else:
            # Standard analytical derivative 
            RF = (2.0 / beta) * np.tan(beta / 2.0)
            dRF_dbeta = (1.0 / (beta * np.cos(beta/2.0)**2)) - (2.0 / (beta**2) * np.tan(beta/2.0))
            dbeta_dcb = -1.0 / np.sin(beta)
            dRF_dcb = dRF_dbeta * dbeta_dcb
            
        # Chain rule mapping of dRF/dcb to the actual input angles
        dRF_di1 = dRF_dcb * dcb_di1
        dRF_di2 = dRF_dcb * dcb_di2
        dRF_da1 = dRF_dcb * dcb_da1
        dRF_da2 = dRF_dcb * dcb_da2

        # Construct the final Jacobian columns using the product rule: 
        # d/dx [ (dl/2) * RF * (v1 + v2) ]
        base_vec = (dl / 2.0) * (v1 + v2)
        J_dl = (RF / 2.0) * (v1 + v2)
        J_i1 = (dl / 2.0) * RF * dv1_di1 + base_vec * dRF_di1
        J_a1 = (dl / 2.0) * RF * dv1_da1 + base_vec * dRF_da1
        J_i2 = (dl / 2.0) * RF * dv2_di2 + base_vec * dRF_di2
        J_a2 = (dl / 2.0) * RF * dv2_da2 + base_vec * dRF_da2
    
    # Scale angular columns back to degree space so the Jacobian matches the input units.
    # Note: J_dl remains unscaled since length is a linear unit.
    rad_to_deg = np.pi / 180.0
    J = np.column_stack([
        J_i1 * rad_to_deg, 
        J_a1 * rad_to_deg, 
        J_i2 * rad_to_deg, 
        J_a2 * rad_to_deg, 
        J_dl               
    ])
        
    return J


# =====================================================================
# STATISTICS & COVARIANCE SIZING
# =====================================================================

def get_confidence_k_factor(
    confidence=0.95,
    dimensions=2,
    scale_mode='sigma',
    sigma_level=2.0,
):
    """
    Return the scale factor used to convert 1-sigma covariance axes to output axes.

    In ``scale_mode='sigma'``, a 2-sigma or 3-sigma EOU uses the same multiplier
    in every dimension. The optional
    ``probability`` mode instead returns the factor for a joint Gaussian coverage
    probability.  These conventions answer different questions and should always
    be labelled explicitly in plots and reports.
    
    Args:
        confidence (float): Joint coverage probability used in probability mode.
        dimensions (int): 1 for an interval, 2 for an ellipse, or 3 for an ellipsoid.
        scale_mode (str): ``'sigma'`` (default) or ``'probability'``.
        sigma_level (float): Axis multiplier used in sigma mode. Defaults to 2.
        
    Returns:
        float: The corresponding scaling factor (k).
    """
    from scipy.stats import chi2, norm

    if dimensions not in (1, 2, 3):
        raise ValueError("Dimensions must be 1, 2, or 3.")

    mode = str(scale_mode).strip().lower()
    if mode == 'sigma':
        if not np.isfinite(sigma_level) or sigma_level <= 0:
            raise ValueError("sigma_level must be a finite positive number.")
        return float(sigma_level)

    if mode not in {'probability', 'coverage'}:
        raise ValueError("scale_mode must be 'sigma' or 'probability'.")
    if not np.isfinite(confidence) or not (0 < confidence < 1):
        raise ValueError("confidence must be a finite number between 0 and 1.")

    if dimensions == 1:
        return float(norm.ppf((1.0 + confidence) / 2.0))
    return float(np.sqrt(chi2.ppf(confidence, dimensions)))

def get_ellipse_parameters(
    C_total_array,
    plane='NE',
    confidence=0.95,
    scale_mode='sigma',
    sigma_level=2.0,
):
    """
    Extracts scale lengths and rotation orientations using vectorized decomposition.
    
    Performs eigenvalue decomposition on sliced covariance matrices to determine 
    the semi-major axes, semi-minor axes, and orientation angles of uncertainty ellipses.

    Args:
        C_total_array (np.ndarray): Array of 3x3 covariance matrices across all stations.
        plane (str): Planar projection plane ('NE', 'NV', 'EV', or '3D'). Defaults to 'NE'.
        confidence (float): Joint coverage probability in probability mode.
        scale_mode (str): ``'sigma'`` (default) or ``'probability'``.
        sigma_level (float): Axis multiplier in sigma mode. Defaults to 2.

    Returns:
        tuple: Depending on the plane, returns either (semi_major, semi_minor, angles) 
               for 2D projections, or (scaled_evalues, evectors) for 3D.
    """
    C_total_array = np.asarray(C_total_array, dtype=float)
    if C_total_array.ndim != 3 or C_total_array.shape[1:] != (3, 3):
        raise ValueError("C_total_array must have shape (n_stations, 3, 3).")
    if len(C_total_array) == 0:
        raise ValueError("C_total_array must contain at least one station.")
    if not np.isfinite(C_total_array).all():
        raise ValueError("C_total_array contains non-finite values.")

    plane = str(plane).upper()
    if plane not in {'NE', 'NV', 'EV', '3D'}:
        raise ValueError("plane must be 'NE', 'NV', 'EV', or '3D'.")

    dimensions = 3 if plane == '3D' else 2
    k = get_confidence_k_factor(
        confidence=confidence,
        dimensions=dimensions,
        scale_mode=scale_mode,
        sigma_level=sigma_level,
    )

    # Slice the 3x3 covariance matrix based on the requested 2D or 3D plane projection
    if plane == 'NE': 
        C_sliced = C_total_array[:, :2, :2]
    elif plane == 'NV': 
        C_sliced = C_total_array[:, [0, 2], :][:, :, [0, 2]]
    elif plane == 'EV': 
        C_sliced = C_total_array[:, 1:3, 1:3]
    else:
        C_sliced = C_total_array

    # Symmetrisation removes harmless round-off asymmetry without hiding a
    # materially non-positive covariance matrix.
    C_sliced = 0.5 * (C_sliced + np.swapaxes(C_sliced, -1, -2))
    evalues, evectors = np.linalg.eigh(C_sliced)
    scale = max(1.0, float(np.max(np.abs(evalues))))
    psd_tolerance = 1e-10 * scale
    min_eig = float(np.min(evalues))
    if min_eig < -psd_tolerance:
        raise ValueError(
            f"Covariance is not positive semidefinite: minimum eigenvalue "
            f"{min_eig:.3e} is below tolerance {-psd_tolerance:.3e}."
        )
    evalues = np.maximum(evalues, 0.0)

    # Return planar ellipse parameters or the three principal ellipsoid axes.
    if plane != '3D':
        semi_minor = k * np.sqrt(evalues[:, 0])
        semi_major = k * np.sqrt(evalues[:, 1])
        angles = np.arctan2(evectors[:, 1, 1], evectors[:, 0, 1])
        return semi_major, semi_minor, angles
    return k * np.sqrt(evalues), evectors
