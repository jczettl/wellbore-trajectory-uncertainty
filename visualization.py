from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from core_math import get_rotation_matrix, get_ellipse_parameters, get_confidence_k_factor


def _uncertainty_label(scale_mode, confidence, sigma_level, region_name):
    """Build an unambiguous label for a plotted uncertainty region."""
    if str(scale_mode).lower() == 'sigma':
        return f"{sigma_level:g}σ {region_name}"
    return f"{confidence * 100:g}% joint {region_name}"

# =====================================================================
# CORE 2D/3D VISUALIZATION SUITE
# =====================================================================

def plot_trajectory_2d(df, C_total_array, plane='NE', ellipse_step=1, confidence=0.95,
                       ns_col='NS', ew_col='EW', tvd_col='TVD',
                       scale_mode='sigma', sigma_level=2.0):
    """
    Generates 2D planar projection profiles detailing trajectory paths and error bounds.
    
    This function projects the global 3x3 covariance matrices down onto a specified 2D 
    structural plane, computing localized normal gradients to draw continuous error 
    envelopes alongside distinct confidence ellipses.

    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe.
        C_total_array (np.ndarray): Array of cumulative 3x3 covariance matrices.
        plane (str): Planar projection plane ('NE', 'NV', or 'EV'). Defaults to 'NE'.
        ellipse_step (int): Station frequency step to plot explicit error ellipses. Defaults to 1.
        confidence (float): Joint coverage probability in probability mode.
        scale_mode (str): ``'sigma'`` (default) or ``'probability'``.
        sigma_level (float): Axis multiplier in sigma mode. Defaults to 2.
        ns_col (str): Column name for North-South coordinates. Defaults to 'NS'.
        ew_col (str): Column name for East-West coordinates. Defaults to 'EW'.
        tvd_col (str): Column name for True Vertical Depth. Defaults to 'TVD'.

    Returns:
        tuple: A pair containing the matplotlib Figure and Axes objects.
    """

    # Extract ellipse scale parameters and orientations for the requested projection plane via decomposition
    semi_major, semi_minor, calc_angles = get_ellipse_parameters(
        C_total_array,
        plane=plane,
        confidence=confidence,
        scale_mode=scale_mode,
        sigma_level=sigma_level,
    )
    
    # Map projection string codes to dataframe columns, labels, axis inversions, and matrix indices
    plane_mappings = {
        'NE': (ew_col, ns_col, 'East (m)', 'North (m)', False, 1, 0),
        'NV': (ns_col, tvd_col, 'North (m)', 'True Vertical Depth (m)', True, 0, 2),
        'EV': (ew_col, tvd_col, 'East (m)', 'True Vertical Depth (m)', True, 1, 2)
    }
    x_col, y_col, x_label, y_label, invert_y, idx_x, idx_y = plane_mappings[plane]
    x_coords, y_coords = df[x_col].values, df[y_col].values
    
    # Extract structural covariance components and compute trajectory path normal gradients
    Cxx = C_total_array[:, idx_x, idx_x]
    Cyy = C_total_array[:, idx_y, idx_y] 
    Cxy = C_total_array[:, idx_x, idx_y]
    dx, dy = np.gradient(x_coords), np.gradient(y_coords)
    ds = np.clip(np.sqrt(dx**2 + dy**2), 1e-10, None)
    nx, ny = -dy / ds, dx / ds
    
    # Calculate positional variance and standard deviation directly along the trajectory's normal vector
    var_n = (nx**2) * Cxx + 2 * nx * ny * Cxy + (ny**2) * Cyy
    k_factor = get_confidence_k_factor(
        confidence=confidence,
        dimensions=2,
        scale_mode=scale_mode,
        sigma_level=sigma_level,
    )
    std_n = np.sqrt(np.clip(var_n, 1e-10, None))
    
    # Compute error envelope spatial offsets for rendering continuous boundary lines
    offset_x = k_factor * (Cxx * nx + Cxy * ny) / std_n
    offset_y = k_factor * (Cxy * nx + Cyy * ny) / std_n
    
    # Initialize the primary matplotlib figure and canvas
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(x_coords + offset_x, y_coords + offset_y, 'r--', lw=1.2, alpha=0.7, label='Error Envelope')
    ax.plot(x_coords - offset_x, y_coords - offset_y, 'r--', lw=1.2, alpha=0.7)
    
    # Create distinct separation masks to differentiate actual physical surveys from interpolated SLERP grid points
    is_interpolated = df.get(
        'is_interpolated',
        np.zeros(len(df), dtype=bool),
    )
    mask_interp = np.asarray(is_interpolated, dtype=bool)
    mask_real = ~mask_interp
    
    # Plot high-density interpolated grid points faintly, and actual survey stations boldly
    ax.scatter(df.loc[mask_interp, x_col], df.loc[mask_interp, y_col], color='blue', s=1, alpha=0.4, label='Interpolated Grid')
    ax.scatter(df.loc[mask_real, x_col], df.loc[mask_real, y_col], color='black', s=2, zorder=5, label='Actual Survey Station')
    
    # Filter station indices based on the requested visual plotting frequency step
    actual_indices = df[mask_real].index.tolist()
    ellipse_indices = [k for idx, k in enumerate(actual_indices) if idx % ellipse_step == 0]
    
    # Render discrete confidence ellipses along the selected survey nodes
    for k in ellipse_indices:
        plot_angle = np.degrees(np.pi/2 - calc_angles[k]) if plane == 'NE' else np.degrees(calc_angles[k])
        for fc, alpha in [('red', 0.12), ('none', 1.0)]:
            ax.add_patch(Ellipse(xy=(x_coords[k], y_coords[k]), width=2*semi_major[k], height=2*semi_minor[k],
                                 angle=plot_angle, edgecolor='red', facecolor=fc, alpha=alpha, linewidth=1.0, zorder=3))

    # --- TIGHT BOUNDING BOX (Rounded to outer 100m for professional map framing) ---
    x_min, x_max = np.min(x_coords - offset_x), np.max(x_coords + offset_x)
    
    if plane in ['NV', 'EV']:
        # Snap the x-axis limits mathematically to the next outer 100m increment
        x_min_rounded = np.floor(x_min / 100.0) * 100.0
        x_max_rounded = np.ceil(x_max / 100.0) * 100.0 + 50
        ax.set_xlim(x_min_rounded, x_max_rounded)
    else:
        # Utilize standard dynamic padding for the Top-Down (NE) map plane
        x_padding = max((x_max - x_min) * 0.15, 50)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)

    # Format the main plot layout and styling presentation
    region_label = _uncertainty_label(
        scale_mode, confidence, sigma_level, 'marginal ellipse'
    )
    ax.set_title(f"Trajectory Profile - {plane} Plane ({region_label})", fontweight='bold')
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.grid(True, linestyle='--', alpha=0.6)
    if invert_y: ax.invert_yaxis()
    ax.set_aspect('auto', adjustable='datalim')
    ax.legend(loc='best') 
    
    return fig, ax

def plot_trajectory_3d_pyvista(df, C_total_array, n_ring_points=32, confidence=0.95,
                               ns_col='NS', ew_col='EW', tvd_col='TVD',
                               inc_col='Incl', azi_col='Azi',
                               scale_mode='sigma', sigma_level=2.0,
                               output_path=None, show=True):
    """
    Loft local cross-hole marginal ellipses along the trajectory.
    
    This tool mathematically transforms global covariance matrices into local 
    cross-sectional coordinate frames (High-Side/Right-Side) via Euler rotation
    matrices.  The resulting surface is a visualization of stationwise 2D
    marginals; it is not a full 3D probability ellipsoid or anti-collision volume.

    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe.
        C_total_array (np.ndarray): Array of cumulative 3x3 covariance matrices.
        n_ring_points (int): Number of circumferential geometric points per error ring. Defaults to 32.
        confidence (float): Joint coverage probability in probability mode.
        scale_mode (str): ``'sigma'`` (default) or ``'probability'``.
        sigma_level (float): Axis multiplier in sigma mode. Defaults to 2.
        ns_col (str): Column name for North. Defaults to 'NS'.
        ew_col (str): Column name for East. Defaults to 'EW'.
        tvd_col (str): Column name for True Vertical Depth. Defaults to 'TVD'.
        inc_col (str): Column name for Inclination. Defaults to 'Incl'.
        azi_col (str): Column name for Azimuth. Defaults to 'Azi'.
        output_path (str or Path, optional): Save a PNG of the initial 3D view.
            Parent directories are created automatically. Defaults to None.
        show (bool): Open the interactive 3D window. Defaults to True.
            If False, render the PNG off-screen without opening a window.

    Returns:
        None: Saves and/or displays the PyVista 3D view as requested.
    """
    if output_path is None and not show:
        return

    import pyvista as pv

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Each ring is a two-dimensional marginal in the plane normal to the hole.
    k_factor = get_confidence_k_factor(
        confidence=confidence,
        dimensions=2,
        scale_mode=scale_mode,
        sigma_level=sigma_level,
    )
    n_stations = len(df)
    rings_3d = np.zeros((n_stations, n_ring_points, 3))
    
    # Generate unit circle coordinates to serve as the template for the cross-sectional rings
    theta = np.linspace(0, 2 * np.pi, n_ring_points)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
    
    # Iterate sequentially through each station to construct local cross-sectional uncertainty rings
    for k in range(n_stations):
        # Fetch the local-to-global rotation matrix for the current station orientation
        R_k = get_rotation_matrix(df[inc_col].iloc[k], df[azi_col].iloc[k])
        
        # Project global covariance into local cross-sectional coordinates (High-Side, Right-Side)
        C_cross = (R_k.T @ C_total_array[k] @ R_k)[:2, :2]
        C_cross = 0.5 * (C_cross + C_cross.T)
        evals, evecs = np.linalg.eigh(C_cross)
        scale = max(1.0, float(np.max(np.abs(evals))))
        if float(np.min(evals)) < -1e-10 * scale:
            raise ValueError(
                f"Cross-hole covariance at station {k} is not positive "
                "semidefinite."
            )
        
        # Build local 3D ellipse points and rotate them back into global spatial coordinates
        ellipse_local_3d = np.vstack([evecs @ np.diag(k_factor * np.sqrt(np.clip(evals, 0, None))) @ unit_circle, np.zeros(n_ring_points)])
        ellipse_global = R_k @ ellipse_local_3d
        
        # Map local uncertainty components to the final global spatial coordinates (East, North, -TVD)
        rings_3d[k, :, 0] = ellipse_global[1, :] + df[ew_col].iloc[k]       # X Axis = East
        rings_3d[k, :, 1] = ellipse_global[0, :] + df[ns_col].iloc[k]       # Y Axis = North
        rings_3d[k, :, 2] = -(ellipse_global[2, :] + df[tvd_col].iloc[k])   # Z Axis = -TVD
        
    # Construct the PyVista structured grid from the stacked ring arrays
    grid = pv.StructuredGrid()
    grid.points = rings_3d.reshape(-1, 3); grid.dimensions = [n_ring_points, n_stations, 1]
    
    # Initialize the PyVista 3D plotter rendering environment
    plotter = pv.Plotter(off_screen=not show, notebook=False)
    plotter.set_background('white')
    
    region_label = _uncertainty_label(
        scale_mode, confidence, sigma_level, 'cross-hole marginal rings'
    )
    plotter.add_title(
        f"3D Trajectory & Cross-Hole Uncertainty Tube ({region_label})",
        font_size=12, 
        color='black'
    )
    
    plotter.add_mesh(
        grid,
        color='red',
        opacity=0.25,
        label=f'{region_label} (visualization)',
    )
    
    # Separate actual physical surveys from interpolated grid points for visualization rendering tiers
    is_interpolated = df.get(
        'is_interpolated',
        np.zeros(len(df), dtype=bool),
    )
    mask_interp = np.asarray(is_interpolated, dtype=bool)
    mask_actual = ~mask_interp
    
    def get_coords(m): 
        return np.column_stack([df.loc[m, ew_col].values, df.loc[m, ns_col].values, -df.loc[m, tvd_col].values])
    
    # Plot distinctly styled survey node points inside the 3D viewer
    if mask_interp.any():
        plotter.add_points(get_coords(mask_interp), color='blue', point_size=2, render_points_as_spheres=True, label='Interpolated Node')
    plotter.add_points(get_coords(mask_actual), color='black', point_size=4, render_points_as_spheres=True, label='Actual Survey')
    
    # Configure bounding grid box and axes geometric representation
    plotter.show_grid(
        xtitle='East (m)',
        ytitle='North (m)',
        ztitle='-TVD (m / Elevation)',
        color='gray'
    )
    
    plotter.add_axes()
    plotter.add_legend(bcolor='white',loc='lower right', size=(0.3, 0.3))
    try:
        if output_path is not None:
            # Render and save before user interaction, so closing the window
            # cannot prevent the initial view from being written to disk.
            plotter.show(interactive=False, auto_close=False)
            plotter.screenshot(str(output_path))
            print(f"Saved plot: {output_path}")
        if show:
            plotter.show(interactive=True)
    finally:
        plotter.close()


# =====================================================================
# DIAGNOSTIC PLOTTING & ANALYTICS
# =====================================================================

def plot_covariance_vs_md(df, C_total_array, md_col='MD', section_borders_md=None):
    """
    Plots the 3x3 covariance matrix elements iteratively against Measured Depth (MD).
    
    Extracts individual variance (main diagonal) and covariance cross-coupling 
    (off-diagonal) terms across North, East, and Vertical spatial components 
    to map their independent growth as a function of depth.

    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe.
        C_total_array (np.ndarray): Array of 3x3 covariance matrices.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        section_borders_md (list, optional): List of measured depths representing structural sections.

    Returns:
        tuple: A pair containing the matplotlib Figure and Axes objects.
    """
    # Filter out interpolated SLERP stations to plot the actual physical variance steps cleanly
    if 'is_interpolated' in df.columns:
        actual_mask = ~df['is_interpolated']
        work_df = df[actual_mask].reset_index(drop=True)
        work_C_array = C_total_array[actual_mask]
    else:
        work_df = df
        work_C_array = C_total_array
    
    md = work_df[md_col].values
    
    # Extract Variances (Main diagonals of the 3x3 covariance matrix)
    nn = work_C_array[:, 0, 0]
    ee = work_C_array[:, 1, 1]
    vv = work_C_array[:, 2, 2]
    
    # Extract Covariances (Off-diagonals, symmetric mathematical pairings)
    ne = work_C_array[:, 0, 1]
    nv = work_C_array[:, 0, 2]
    ev = work_C_array[:, 1, 2]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot isolated variance profiles
    ax.plot(md, nn, label='NN (Var North)', color='#1f77b4', lw=2)
    ax.plot(md, ee, label='EE (Var East)', color='#2ca02c', lw=2)
    ax.plot(md, vv, label='VV (Var Vertical)', color='#d62728', lw=2)
    
    # Plot covariance cross-coupling profiles
    ax.plot(md, ne, label='NE (Cov North-East)', color='#ff7f0e', linestyle='--', lw=1.5)
    ax.plot(md, nv, label='NV (Cov North-Vertical)', color='#9467bd', linestyle='--', lw=1.5)
    ax.plot(md, ev, label='EV (Cov East-Vertical)', color='#8c564b', linestyle='--', lw=1.5)

    # Inject vertical section borders to aid operational context mapping
    if section_borders_md is not None:
        for idx, border in enumerate(section_borders_md):
            # Only append a string label to the first element to avoid duplicate legend entries
            label = 'Section Border' if idx == 0 else None
            ax.axvline(border, color='black', linestyle='-', lw=2.5, alpha=0.75, zorder=1, label=label)
    
    ax.set_title('Covariance Matrix Elements vs Measured Depth', fontweight='bold')
    ax.set_xlabel('Measured Depth (m)')
    ax.set_ylabel('Covariance Value (m²)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    return fig, ax

def investigate_covariance_spikes(df, C_total_array, dls_interval=30.0, 
                                  md_col='MD', inc_col='Incl', azi_col='Azi',
                                  use_state_space_propagation=False,
                                  section_borders_md=None):
    """
    Diagnostic dashboard mapping spatial variance growth alongside wellbore kinematics.

    Combines targeted mathematical error diagnostics with a general trajectory overview. 

    Dashboard Panels:
      1. Normalized Error Growth: Covariance Trace per meter.
      2-4. Inclination, Azimuth, DLS: Kinematic trajectory overview.
      5. Interval Length: Highlights survey gaps.

    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe.
        C_total_array (np.ndarray): Array of cumulative 3x3 covariance matrices.
        dls_interval (float): Standard interval length for Dogleg Severity calculation. Defaults to 30.0.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        inc_col (str): Column name for Inclination. Defaults to 'Incl'.
        azi_col (str): Column name for Azimuth. Defaults to 'Azi'.
        use_state_space_propagation (bool): Flag indicating if high-density grid is actively utilized. Defaults to False.
        section_borders_md (list, optional): Depths representing discrete structural/drilling sections.

    Returns:
        tuple: A pair containing the matplotlib Figure and Axes objects.
    """
    # Filter operational dataset based on interpolation mode and physical station type
    if not use_state_space_propagation and 'is_interpolated' in df.columns:
        actual_mask = ~df['is_interpolated']
        work_df = df[actual_mask].reset_index(drop=True)
        work_C_array = C_total_array[actual_mask]
    else:
        work_df = df.copy()
        work_C_array = C_total_array.copy()
        
    md = work_df[md_col].values
    inc = work_df[inc_col].values 
    azi = work_df[azi_col].values 
    dls = work_df['DLS'].values
    n_stations = len(md)
    
    # Calculate Total Spatial Variance (Trace of the 3x3 covariance matrix: NN + EE + VV)
    total_variance = np.trace(work_C_array, axis1=1, axis2=2)
    
    # Compute absolute step-wise change in mathematical variance across consecutive stations
    delta_variance = np.zeros(n_stations)
    delta_variance[1:] = np.diff(total_variance)
    
    # Compute physical interval step lengths (dl)
    dl = np.zeros(n_stations)
    dl[1:] = np.diff(md)
    
    # Normalize error growth rate safely by step length (Variance injected per Meter)
    variance_growth_rate = np.zeros(n_stations)
    valid_dl = dl > 1e-4
    variance_growth_rate[valid_dl] = delta_variance[valid_dl] / dl[valid_dl]
            
    # Initialize 5-row diagnostic subplot stack layout 
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Normalized Error Growth Rate (Highlights error sensitivity spikes)
    axes[0].plot(md, variance_growth_rate, color='red', lw=1.5, marker='o', markersize=3, alpha=0.8)
    axes[0].set_title('Covariance-trace change per unit measured depth', fontweight='bold')
    axes[0].set_ylabel('Variance per Meter (m²/m)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Inclination
    axes[1].plot(md, inc, color='darkorange', lw=1.5, marker='d', markersize=3, alpha=0.8)
    axes[1].set_title('Inclination', fontweight='bold')
    axes[1].set_ylabel('Inclination (°)')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Azimuth
    axes[2].plot(md, azi, color='purple', lw=1.5, marker='v', markersize=3, alpha=0.8)
    axes[2].set_title('Azimuth', fontweight='bold')
    axes[2].set_ylabel('Azimuth (°)')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 4: Dogleg Severity
    axes[3].plot(md, dls, color='green', lw=1.5, marker='^', markersize=3, alpha=0.8)
    axes[3].set_title(f'Dogleg Severity (DLS per {dls_interval}m)', fontweight='bold')
    axes[3].set_ylabel('DLS (°/30m)')
    axes[3].grid(True, linestyle='--', alpha=0.6)

    # Plot 5: Physical Interval Lengths
    axes[4].plot(md, dl, color='blue', lw=1.5, marker='s', markersize=3, alpha=0.8)
    axes[4].set_title('Interval Length ($\\Delta$MD)', fontweight='bold')
    axes[4].set_ylabel('Step Length (m)')
    axes[4].set_xlabel('Measured Depth (m)')
    axes[4].grid(True, linestyle='--', alpha=0.6)
    
    # --- HIGHLIGHTS AND BORDERS ---
    for ax in axes:
        # 1. Overlay faint dotted lines denoting actual physical survey measurements
        if 'is_interpolated' in df.columns:
            actual_mds = df.loc[~df['is_interpolated'], md_col].values
            for a_md in actual_mds:
                ax.axvline(a_md, color='gray', linestyle=':', alpha=0.3, lw=1, zorder=0)
        
        # 2. Emphasize operational section border lines strictly on top of data
        if section_borders_md is not None:
            for border in section_borders_md:
                ax.axvline(border, color='black', linestyle='-', lw=2.5, alpha=0.85, zorder=2)
                
    fig.tight_layout()
    return fig, axes
