import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# DIAGNOSTIC & KINEMATIC BREAKDOWNS
# =====================================================================

def plot_ev_covariance_breakdown(df, C_total_array, term_covariances, error_model, md_col='MD'):
    """
    Deconstructs the East-Vertical (EV) covariance element into its root ISCWSA error sources.
    
    This diagnostic tool groups individual error covariance contributions strictly by their 
    primary sensory measurement axis: Measured Depth (MD), Inclination (Inc), or Azimuth (Azi). 
    By stacking these grouped sums over structural annotations, it visually demonstrates 
    which physical sensor inaccuracies are responsible for driving spatial uncertainty 
    within specific wellbore intervals.
    
    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe.
        C_total_array (np.ndarray): Total cumulative 3x3 covariance array.
        term_covariances (dict): Dictionary mapping individual ISCWSA term strings to their covariance arrays.
        error_model (dict): Dictionary defining the ISCWSA structural parameters for each term.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        
    Returns:
        matplotlib.figure.Figure: The finalized matplotlib Figure object.
    """
    # Extract Measured Depth telemetry
    md = df[md_col].values
    
    # Isolate the total cumulative EV component (East mapping to index 1, Vertical to index 2)
    ev_total = C_total_array[:, 1, 2]
    
    # Initialize zero-arrays to serve as accumulators for the sensory axis groupings
    ev_md_sum = np.zeros(len(md))
    ev_inc_sum = np.zeros(len(md))
    ev_azi_sum = np.zeros(len(md))
    
    # Setup the primary plot dimensions
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Lay down the Total EV curve directly in the background as a semi-transparent anchor
    ax.plot(md, ev_total, color='black', lw=8, alpha=0.15, label='TOTAL EV Covariance')
    
    # Loop over every active ISCWSA error term within the operational dictionary
    for term, C_term in term_covariances.items():
        # Slice out the specific EV off-diagonal component for this individual term
        ev_term = C_term[:, 1, 2]
        
        # Interrogate the error model to resolve this term's base measurement axis
        axis = error_model[term]['axis']
        
        # Accumulate the term's variance strictly into its designated axis bucket
        if axis == 'MD':
            ev_md_sum += ev_term
        elif axis == 'Inc':
            ev_inc_sum += ev_term
        elif axis == 'Azi':
            ev_azi_sum += ev_term
            
        # Draw the isolated term as a fine dashed line to reveal subtle substructure
        ax.plot(md, ev_term, lw=1.5, alpha=0.7, linestyle='--', label=f'{term} ({axis})')

    # Add Structural Annotations based on precise kinematic inflection points
    ax.axvspan(0, 620.0, color='gray', alpha=0.1, label='Section 1: Near-Vertical')
    ax.axvspan(620.0, 2130.0, color='blue', alpha=0.05, label='Section 2: 3D Turn')
    ax.axvspan(2130.0, 3168.1, color='orange', alpha=0.05, label='Section 3: High-Inclination Build')
    ax.axvspan(3168.1, 3571.7, color='purple', alpha=0.05, label='Section 4: Inclination Drop')
    ax.axvspan(3571.7, md[-1], color='green', alpha=0.05, label='Section 5: 40-Degree Hold')
        
    # Superimpose the accumulated grouped sums as distinct, bold solid lines
    ax.plot(md, ev_md_sum, color='#1f77b4', lw=3, label='SUM: MD Terms')
    ax.plot(md, ev_inc_sum, color='#d62728', lw=3, label='SUM: Inc Terms')
    ax.plot(md, ev_azi_sum, color='#2ca02c', lw=3, label='SUM: Azi Terms')
    
    # Formal aesthetics, titling, and Cartesian grid formatting
    ax.set_title('EV (East-Vertical) Covariance Component Breakdown vs Measured Depth', fontweight='bold', fontsize=14)
    ax.set_xlabel('Measured Depth (m)', fontsize=12)
    ax.set_ylabel('EV Covariance Value (m²)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Displace the legend completely outside the plot body to prevent visual obstruction
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    return fig

def plot_sage_diagnostic(df, term_covariances, md_col='MD'):
    """
    Isolates the East Variance (EE), Vertical Variance (VV), and East-Vertical 
    Covariance (EV) elements generated exclusively by the SAGE (Sag Effect) error term.
    
    This diagnostic shows how a systematic, inclination-dependent term changes
    with well geometry, including the sign of its EV covariance contribution.
    
    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe.
        term_covariances (dict): Dictionary mapping individual ISCWSA term strings to their covariance arrays.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        
    Returns:
        matplotlib.figure.Figure: The finalized matplotlib Figure object.
    """
    # Safeguard: Verify the SAGE term is actively present in the extracted dictionary
    if 'SAGE' not in term_covariances:
        print("[!] SAGE term not found in covariance dictionary.")
        return
        
    md = df[md_col].values
    C_sage = term_covariances['SAGE']
    
    # Isolate the targeted SAGE spatial covariance elements
    ee_sage = C_sage[:, 1, 1]  # East Variance
    vv_sage = C_sage[:, 2, 2]  # Vertical Variance
    ev_sage = C_sage[:, 1, 2]  # East-Vertical Covariance
    
    # Setup the primary diagnostic plot canvas
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the isolated mathematical curves
    ax.plot(md, ee_sage, color='#2ca02c', lw=3, label='SAGE EE (East Variance)')
    ax.plot(md, vv_sage, color='#d62728', lw=3, label='SAGE VV (Vertical Variance)')
    ax.plot(md, ev_sage, color='#8c564b', lw=3, linestyle='--', label='SAGE EV (East-Vertical Covariance)')
    
    # Volve-specific trajectory sections used in the report.
    ax.axvspan(0, 620.0, color='gray', alpha=0.1, label='Section 1: Near-Vertical')
    ax.axvspan(620.0, 2130.0, color='blue', alpha=0.05, label='Section 2: 3D Turn')
    ax.axvspan(2130.0, 3168.1, color='orange', alpha=0.05, label='Section 3: High-Inclination Build')
    ax.axvspan(3168.1, 3571.7, color='purple', alpha=0.05, label='Section 4: Inclination Drop')
    ax.axvspan(3571.7, md[-1], color='green', alpha=0.05, label='Section 5: 40-Degree Hold')
    
    # Formal aesthetics, titling, and Cartesian grid formatting
    ax.set_title('SAGE Covariance Diagnostic vs Measured Depth', fontweight='bold', fontsize=14)
    ax.set_xlabel('Measured Depth (m)', fontsize=12)
    ax.set_ylabel('Covariance Value (m²)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    return fig

def plot_sage_raw_vectors(df, term_diagnostics, md_col='MD'):
    """
    Visualizes the raw, unsquared physical error vectors (East and TVD) specifically 
    for the SAGE error term.
    
    The complete station vector includes the current endpoint contribution. Its
    East--Vertical component product therefore matches the SAGE EV covariance at
    the same station.
    
    Args:
        df (pd.DataFrame): Wellbore trajectory dataframe.
        term_diagnostics (dict): Dictionary containing the un-squared accumulator vectors.
        md_col (str): Column name for Measured Depth. Defaults to 'MD'.
        
    Returns:
        matplotlib.figure.Figure: The finalized matplotlib Figure object.
    """
    # Safeguard: Verify the SAGE term is actively present in the extracted dictionary
    if 'SAGE' not in term_diagnostics:
        print("[!] SAGE term not found in diagnostic dictionary.")
        return
        
    if 'e_total' not in term_diagnostics['SAGE']:
        raise KeyError("SAGE diagnostics do not contain the complete 'e_total' vector.")

    # The identity is evaluated at physical survey stations. Interpolated
    # covariances are drawing aids and are not outer products of interpolated
    # error vectors.
    if 'is_interpolated' in df.columns:
        mask = ~df['is_interpolated'].to_numpy(dtype=bool)
    else:
        mask = np.ones(len(df), dtype=bool)

    md = df.loc[mask, md_col].to_numpy()
    # Index 1 is East and index 2 is TVD in the NEV frame.
    e_total_sage = term_diagnostics['SAGE']['e_total'][mask]
    e_east = e_total_sage[:, 1]
    e_tvd = e_total_sage[:, 2]
    
    # For a correlated source C = E E^T, hence C_EV = E_E E_V.
    ev_product = e_east * e_tvd
    
    # Setup the primary diagnostic plot canvas
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot the raw un-squared physical accumulation vectors
    ax1.plot(md, e_east, color='#2ca02c', lw=3, label='SAGE $e_E$ (Raw East Vector)')
    ax1.plot(md, e_tvd, color='#d62728', lw=3, label='SAGE $e_V$ (Raw TVD Vector)')
    
    # Overlay the resulting EV Covariance product as a dotted line for direct validation
    ax1.plot(md, ev_product, color='#8c564b', lw=2.5, linestyle=':', alpha=0.8, 
             label='SAGE EV Covariance = ($e_E \\times e_V$)')
    
    # Add a zero reference line to show vector sign changes.
    ax1.axhline(0, color='black', lw=2, linestyle='-', alpha=0.8)
    
    # Add Structural Annotations based on precise kinematic inflection points
    ax1.axvspan(0, 620.0, color='gray', alpha=0.1, label='Section 1: Near-Vertical')
    ax1.axvspan(620.0, 2130.0, color='blue', alpha=0.05, label='Section 2: 3D Turn')
    ax1.axvspan(2130.0, 3168.1, color='orange', alpha=0.05, label='Section 3: High-Inclination Build')
    ax1.axvspan(3168.1, 3571.7, color='purple', alpha=0.05, label='Section 4: Inclination Drop')
    ax1.axvspan(3571.7, md[-1], color='green', alpha=0.05, label='Section 5: 40-Degree Hold')
    
    # Formal aesthetics, titling, and Cartesian grid formatting
    ax1.set_title(r'SAGE Complete Error Vector Diagnostic ($\mathbf{E}_E$ and $\mathbf{E}_V$)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Measured Depth (m)', fontsize=12)
    ax1.set_ylabel('Raw Error Accumulation (m)  /  Covariance (m²)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    return fig
