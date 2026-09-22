import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =====================================================================
# INTERACTIVE VISUALIZATION DASHBOARDS
# =====================================================================

def plot_interactive_dashboard(df, C_total_array, md_col='MD', ns_col='NS', ew_col='EW', tvd_col='TVD'):
    """
    Creates an interactive visualization dashboard integrating 3D spatial trajectory 
    projections with dynamic covariance matrix evolution.
    
    This pedagogical tool maps error accumulation (uncertainty) directly to the 
    physical wellbore path. Hovering over the covariance plot dynamically projects 
    the location onto the 2D spatial trajectory plots (Top-Down and Cross-Sections) 
    and displays the exact Measured Depth (MD) and True Vertical Depth (TVD) values 
    in a text box.
    
    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe containing spatial survey nodes.
        C_total_array (np.ndarray): Array of 3x3 covariance matrices corresponding to each station.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        ns_col (str): Column name for North. Defaults to 'NS'.
        ew_col (str): Column name for East. Defaults to 'EW'.
        tvd_col (str): Column name for True Vertical Depth. Defaults to 'TVD'.
        
    Returns:
        matplotlib.figure.Figure: The constructed interactive matplotlib Figure object.
    """
    
    # Strip out artificially interpolated stations to ensure we only visualize 
    # the variance steps occurring at actual physical survey nodes
    if 'is_interpolated' in df.columns:
        actual_mask = ~df['is_interpolated']
        work_df = df[actual_mask].reset_index(drop=True)
        work_C_array = C_total_array[actual_mask]
    else:
        work_df = df
        work_C_array = C_total_array

    # Extract spatial coordinates into flat NumPy arrays for plotting performance
    md = work_df[md_col].values
    ns = work_df[ns_col].values
    ew = work_df[ew_col].values
    tvd = work_df[tvd_col].values
    
    # Extract Variances (Main Diagonals) and Covariances (Off-Diagonals)
    nn = work_C_array[:, 0, 0]
    ee = work_C_array[:, 1, 1]
    vv = work_C_array[:, 2, 2]
    ne = work_C_array[:, 0, 1]
    nv = work_C_array[:, 0, 2]
    ev = work_C_array[:, 1, 2]
    
    # Initialize the primary Figure and GridSpec Layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
    
    # Instantiate Subplots mapped to the GridSpec
    ax_ne = fig.add_subplot(gs[0, 0])
    ax_nv = fig.add_subplot(gs[0, 1])
    ax_ev = fig.add_subplot(gs[0, 2])
    ax_cov = fig.add_subplot(gs[1, :]) # Spans across all 3 columns
    
    # =====================================================================
    # ROW 1: SPATIAL 2D TRAJECTORY PROJECTIONS
    # =====================================================================
    
    # NE Plane (Top-Down Map View)
    ax_ne.plot(ew, ns, 'k-', lw=1.5, label='Trajectory')
    ax_ne.set_title("NE Plane (Top-Down)", fontweight='bold')
    ax_ne.set_xlabel("East (m)")
    ax_ne.set_ylabel("North (m)")
    ax_ne.grid(True, linestyle='--', alpha=0.6)
    
    # NV Plane (North-South Cross-Section)
    ax_nv.plot(ns, tvd, 'k-', lw=1.5)
    ax_nv.set_title("NV Plane", fontweight='bold')
    ax_nv.set_xlabel("North (m)")
    ax_nv.set_ylabel("True Vertical Depth (m)")
    ax_nv.invert_yaxis() 
    ax_nv.grid(True, linestyle='--', alpha=0.6)
    
    # EV Plane (East-West Cross-Section)
    ax_ev.plot(ew, tvd, 'k-', lw=1.5)
    ax_ev.set_title("EV Plane", fontweight='bold')
    ax_ev.set_xlabel("East (m)")
    ax_ev.set_ylabel("True Vertical Depth (m)")
    ax_ev.invert_yaxis()
    ax_ev.grid(True, linestyle='--', alpha=0.6)
    
    # =====================================================================
    # ROW 2: COVARIANCE MATRIX EVOLUTION
    # =====================================================================
    
    ax_cov.plot(md, nn, label='NN (Var North)', color='#1f77b4', lw=2)
    ax_cov.plot(md, ee, label='EE (Var East)', color='#2ca02c', lw=2)
    ax_cov.plot(md, vv, label='VV (Var Vertical)', color='#d62728', lw=2)
    ax_cov.plot(md, ne, label='NE (Cov North-East)', color='#ff7f0e', linestyle='--', lw=1.5)
    ax_cov.plot(md, nv, label='NV (Cov North-Vertical)', color='#9467bd', linestyle='--', lw=1.5)
    ax_cov.plot(md, ev, label='EV (Cov East-Vertical)', color='#8c564b', linestyle='--', lw=1.5)
    
    ax_cov.set_title('Covariance Matrix Elements vs Measured Depth', fontweight='bold')
    ax_cov.set_xlabel('Measured Depth (MD)')
    ax_cov.set_ylabel('Covariance Value (m²)')
    ax_cov.grid(True, linestyle='--', alpha=0.6)
    ax_cov.legend(loc='upper left')
    
    # =====================================================================
    # INTERACTIVE EVENT HANDLING
    # =====================================================================
    
    # Initialize dynamic markers and indicators (invisible/default states)
    line_cov = ax_cov.axvline(x=md[0], color='black', linestyle=':', lw=2, alpha=0.7)
    point_ne, = ax_ne.plot([ew[0]], [ns[0]], 'ro', markersize=8, zorder=5)
    point_nv, = ax_nv.plot([ns[0]], [tvd[0]], 'ro', markersize=8, zorder=5)
    point_ev, = ax_ev.plot([ew[0]], [tvd[0]], 'ro', markersize=8, zorder=5)
    
    # Construct the dynamic information bounding box
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    text_box = ax_cov.text(0.02, 0.10, '', transform=ax_cov.transAxes, 
                           fontsize=11, verticalalignment='bottom', bbox=props, fontweight='bold')
    
    # Hover Event Callback
    def on_mouse_move(event):
        # Only process calculations if the cursor is actively within the covariance axis bounds
        if event.inaxes == ax_cov and event.xdata is not None:
            # Locate the nearest valid physical survey station to the cursor's continuous X position
            idx = np.abs(md - event.xdata).argmin()
            current_md = md[idx]
            
            # Snap the intersecting reference line to the resolved MD
            line_cov.set_xdata([current_md, current_md])
            
            # Project the spatial position onto all three 2D structural planes
            point_ne.set_data([ew[idx]], [ns[idx]])
            point_nv.set_data([ns[idx]], [tvd[idx]])
            point_ev.set_data([ew[idx]], [tvd[idx]])
            
            # Update the text box payload with exact positional telemetry
            text_box.set_text(f"Depth Locator:\nMD: {current_md:.1f} m\nTVD: {tvd[idx]:.1f} m")
            
            # Queue a highly efficient UI redraw event
            fig.canvas.draw_idle()

    # Bind the hovering event handler directly to the matplotlib figure canvas
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    # Force an immediate event cycle to initialize the text box at the surface tie-in point
    on_mouse_move(type('Event', (object,), {'inaxes': ax_cov, 'xdata': md[0]})())
    return fig


# =====================================================================
# STATIC VISUALIZATION DASHBOARDS
# =====================================================================

def plot_static_dashboard(df, C_total_array, md_col='MD', ns_col='NS', ew_col='EW', tvd_col='TVD', section_borders_md=None):
    """
    Constructs a comprehensive static dashboard mapping 3D trajectory profiles 
    against accumulated error covariances.
    
    The trajectory is optionally divided into distinct kinematic sections designated 
    by Measured Depth boundary markers. This enables the direct visual correlation 
    between physical drilling behaviors (e.g., turning, building) and their 
    resultant covariance responses.
    
    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe.
        C_total_array (np.ndarray): Array of 3x3 cumulative covariance matrices.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        ns_col (str): Column name for North. Defaults to 'NS'.
        ew_col (str): Column name for East. Defaults to 'EW'.
        tvd_col (str): Column name for True Vertical Depth. Defaults to 'TVD'.
        section_borders_md (list, optional): List of Measured Depths to draw segment boundary lines.
        
    Returns:
        matplotlib.figure.Figure: The constructed static matplotlib Figure object.
    """
    
    # Strip out artificially interpolated stations to ensure we only visualize 
    # the variance steps occurring at actual physical survey nodes
    if 'is_interpolated' in df.columns:
        actual_mask = ~df['is_interpolated']
        work_df = df[actual_mask].reset_index(drop=True)
        work_C_array = C_total_array[actual_mask]
    else:
        work_df = df
        work_C_array = C_total_array

    # Extract spatial coordinates into flat NumPy arrays for plotting performance
    md = work_df[md_col].values
    ns = work_df[ns_col].values
    ew = work_df[ew_col].values
    tvd = work_df[tvd_col].values
    
    # Extract Variances (Main Diagonals) and Covariances (Off-Diagonals)
    nn = work_C_array[:, 0, 0]
    ee = work_C_array[:, 1, 1]
    vv = work_C_array[:, 2, 2]
    ne = work_C_array[:, 0, 1]
    nv = work_C_array[:, 0, 2]
    ev = work_C_array[:, 1, 2]
    
    # Initialize the primary Figure and GridSpec Layout
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)
    
    # Instantiate Subplots mapped to the GridSpec
    ax_ne = fig.add_subplot(gs[0, 0])
    ax_nv = fig.add_subplot(gs[0, 1])
    ax_ev = fig.add_subplot(gs[0, 2])
    ax_cov = fig.add_subplot(gs[1, :]) # Spans across all 3 columns
    
    # --- Resolve Boundary Indices ---
    boundary_indices = []
    if section_borders_md is not None:
        for b in section_borders_md:
            # Restrict boundary placement to the actual extent of the wellbore to prevent out-of-bounds indexing errors
            if md[0] <= b <= md[-1]:
                boundary_indices.append(np.abs(md - b).argmin())

    # ---------------------------------------------------------------------
    # ROW 1: SPATIAL 2D TRAJECTORY PROJECTIONS
    # ---------------------------------------------------------------------
    # Plot the continuous global wellbore trajectory onto the three primary planes
    ax_ne.plot(ew, ns, color='black', lw=1.5, label='Wellbore Trajectory')
    ax_nv.plot(ns, tvd, color='black', lw=1.5)
    ax_ev.plot(ew, tvd, color='black', lw=1.5)

    # Inject a hidden proxy artist specifically to generate the legend entry for the boundary markers
    if boundary_indices:
        ax_ne.plot([], [], marker='x', color='red', linestyle='None', markersize=6, label='Section Boundaries')

    # --- Draw Kinematic Boundary Markers ---
    for idx in boundary_indices:
        # Stamp 'x' markers onto all 2D spatial trajectory plots
        ax_ne.plot(ew[idx], ns[idx], marker='x', color='red', markersize=8, markeredgewidth=2, zorder=5)
        ax_nv.plot(ns[idx], tvd[idx], marker='x', color='red', markersize=8, markeredgewidth=2, zorder=5)
        ax_ev.plot(ew[idx], tvd[idx], marker='x', color='red', markersize=8, markeredgewidth=2, zorder=5)
        
        # Draw corresponding vertical tracking lines down the length of the Covariance plot
        ax_cov.axvline(x=md[idx], color='black', linestyle='-', lw=2.5, alpha=0.75, zorder=3)

    # Finalize labels, formatting, and standard NED orientation grids
    ax_ne.set_title("NE Plane (Top-Down)", fontweight='bold')
    ax_ne.set_xlabel("East (m)")
    ax_ne.set_ylabel("North (m)")
    ax_ne.grid(True, linestyle='--', alpha=0.6)
    ax_ne.legend(loc='best')
    
    ax_nv.set_title("NV Plane", fontweight='bold')
    ax_nv.set_xlabel("North (m)")
    ax_nv.set_ylabel("True Vertical Depth (m)")
    ax_nv.invert_yaxis() 
    ax_nv.grid(True, linestyle='--', alpha=0.6)
    
    ax_ev.set_title("EV Plane", fontweight='bold')
    ax_ev.set_xlabel("East (m)")
    ax_ev.set_ylabel("True Vertical Depth (m)")
    ax_ev.invert_yaxis()
    ax_ev.grid(True, linestyle='--', alpha=0.6)
    
    # ---------------------------------------------------------------------
    # ROW 2: COVARIANCE MATRIX EVOLUTION
    # ---------------------------------------------------------------------
    ax_cov.plot(md, nn, label='NN (Var North)', color='#1f77b4', lw=2)
    ax_cov.plot(md, ee, label='EE (Var East)', color='#2ca02c', lw=2)
    ax_cov.plot(md, vv, label='VV (Var Vertical)', color='#d62728', lw=2)
    ax_cov.plot(md, ne, label='NE (Cov North-East)', color='#ff7f0e', linestyle='--', lw=1.5)
    ax_cov.plot(md, nv, label='NV (Cov North-Vertical)', color='#9467bd', linestyle='--', lw=1.5)
    ax_cov.plot(md, ev, label='EV (Cov East-Vertical)', color='#8c564b', linestyle='--', lw=1.5)
    
    ax_cov.set_title('Covariance Matrix Elements vs Measured Depth', fontweight='bold')
    ax_cov.set_xlabel('Measured Depth (m)')
    ax_cov.set_ylabel('Covariance Value (m²)')
    ax_cov.grid(True, linestyle='--', alpha=0.6)
    ax_cov.legend(loc='upper left', ncol=2)
    
    return fig


