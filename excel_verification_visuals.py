import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================
_COMPONENT_MAP = {
    "NN": (0, 0), "EE": (1, 1), "VV": (2, 2), 
    "NE": (0, 1), "NV": (0, 2), "EV": (1, 2)
}

# =====================================================================
# VISUAL DEBUGGING & DIAGNOSTICS SUITE
# =====================================================================

def plot_tvd_validation(df_out, remove_baseline=False, save_filename=None):
    """
    Plots the absolute difference between Calculated and Excel Reference TVD,
    and dynamically shades geometric hold sections (where Delta Inc = 0, Delta Azi = 0).

    Args:
        df_out (pd.DataFrame): Dataframe containing 'MD', 'TVD', 'Original_TVD', 'Incl', and 'Azi'.
        remove_baseline (bool): Subtracts mean difference to show pure fluctuations.
        save_filename (str, optional): If provided, saves the plot to this filepath.
    """
    md = df_out["MD"].to_numpy()
    calc = df_out["TVD"].to_numpy()
    excel = df_out["Original_TVD"].to_numpy()
    
    diff = calc - excel
    baseline = np.mean(diff)
    fluct = diff - baseline if remove_baseline else diff

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(md, fluct, lw=2, label='TVD Deviation', zorder=3)
    ax.axhline(0, color="k", ls="--", zorder=2)
    
    # DYNAMIC SHADING: Detect and shade hold sections
    if "Incl" in df_out.columns and "Azi" in df_out.columns:
        inc = df_out["Incl"].to_numpy()
        azi = df_out["Azi"].to_numpy()
        
        delta_inc = np.abs(np.diff(inc))
        delta_azi = np.abs(np.diff(azi))
        
        # A step is a kinematic hold if the angular change is mathematically zero
        is_hold = (delta_inc < 1e-8) & (delta_azi < 1e-8)
        
        in_hold, start_md, added_label = False, None, False
        
        for k in range(len(is_hold)):
            if is_hold[k] and not in_hold:
                in_hold = True
                start_md = md[k]
            elif not is_hold[k] and in_hold:
                in_hold = False
                end_md = md[k]
                label = 'Hold Section ($\\Delta I=0, \\Delta A=0$)' if not added_label else None
                ax.axvspan(start_md, end_md, color='gray', alpha=0.2, label=label, zorder=1)
                added_label = True
                
        if in_hold:
            label = 'Hold Section ($\\Delta I=0, \\Delta A=0$)' if not added_label else None
            ax.axvspan(start_md, md[-1], color='gray', alpha=0.2, label=label, zorder=1)
    
    ax.set_xlabel("Measured Depth (m)")
    ax.set_ylabel("ΔTVD")
    ax.set_title("TVD Difference (Calculated − Excel)")
    ax.grid(True)
    ax.legend(loc='upper left')

    print("\n[TVD Statistics]")
    print(f"Mean Difference : {baseline:.6e}")
    print(f"Std Deviation   : {np.std(fluct):.6e}")
    print(f"Maximum Error   : {np.max(np.abs(fluct)):.6e}")
    print(f"RMSE            : {np.sqrt(np.mean(fluct**2)):.6e}")

    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"-> Saved plot as {save_filename}")
    plt.show()

def plot_covariance_validation(df_out, calc_covs, excel_covs, term, remove_baseline=False):
    """
    Plots absolute errors for NN, NE, NV, EV, EE, and VV components side-by-side.

    Args:
        df_out (pd.DataFrame): Dataframe containing 'MD' column.
        calc_covs (dict): Calculated covariance arrays.
        excel_covs (dict): Reference covariance DataFrames.
        term (str): Target ISCWSA term to plot.
        remove_baseline (bool): If true, shifts the graph to ignore static offsets.
    """
    components = ["NN", "NE", "NV", "EV", "EE", "VV"]
    md = df_out["MD"].to_numpy()

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 2)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    axes.append(fig.add_subplot(gs[3, :])) # Bottom spanning row

    norm = None
    print(f"\n[{term} Covariance Validation Stats]")
    
    for ax, comp in zip(axes[:6], components):
        i, j = _COMPONENT_MAP[comp]
        calc = calc_covs[term][:, i, j]
        excel = excel_covs[term][comp].to_numpy()
        
        n = min(len(calc), len(excel))
        diff = calc[:n] - excel[:n]
        baseline = np.mean(diff)
        if remove_baseline: diff -= baseline
            
        norm = diff**2 if norm is None else norm + diff**2
            
        ax.plot(md[:n], diff, lw=2)
        ax.axhline(0, color="k", ls="--")
        ax.set_title(comp)
        ax.set_ylabel("Difference")
        ax.grid(True)
        print(f"  {comp} -> Mean: {baseline:.3e} | Std: {np.std(diff):.3e} | Max: {np.max(np.abs(diff)):.3e}")

    norm = np.sqrt(norm)
    axes[6].plot(md[:len(norm)], norm, color="red", lw=2)
    axes[6].set_title("Euclidean Covariance Error")
    axes[6].set_xlabel("Measured Depth (m)")
    axes[6].set_ylabel("||ΔC||")
    axes[6].grid(True)

    plt.suptitle(f"{term} Covariance Absolute Difference Validation")
    plt.tight_layout()
    plt.show()

def plot_covariance_scaling(df_out, calc_covs, excel_covs, term):
    """
    Plots the relative scaling error (Truth / Calculated) for a given error term.
    Args:
            df_out (pd.DataFrame): Dataframe containing 'MD' column.
            calc_covs (dict): Calculated covariance arrays.
            excel_covs (dict): Reference covariance DataFrames.
            term (str): Target ISCWSA term to plot.
    """
    components = ["NN", "NE", "NV", "EV", "EE", "VV"]
    md = df_out["MD"].to_numpy()

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 2)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    axes.append(fig.add_subplot(gs[3, :]))

    norm = None
    print(f"\n[{term} Covariance Scaling Stats]")
    
    for ax, comp in zip(axes[:6], components):
        i, j = _COMPONENT_MAP[comp]
        calc = calc_covs[term][:, i, j]
        truth = excel_covs[term][comp].to_numpy()
        
        n = min(len(calc), len(truth))
        calc, truth = calc[:n], truth[:n]
        
        scale = np.full(n, np.nan)
        mask = np.abs(calc) > 1e-12
        scale[mask] = truth[mask] / calc[mask]

        deviation = scale - 1.0
        norm = deviation**2 if norm is None else norm + deviation**2

        ax.plot(md[:n], scale, lw=2)
        ax.axhline(1.0, color="k", linestyle="--")
        ax.set_title(comp)
        ax.set_ylabel("Truth / Calc")
        ax.grid(True)

        valid = scale[mask]
        print(f"  {comp} -> Mean Scale: {np.mean(valid):.6f} | Std: {np.std(valid):.6e}")

    norm = np.sqrt(norm)
    axes[6].plot(md[:len(norm)], norm, color="red", lw=2)
    axes[6].set_title("Euclidean Scaling Error")
    axes[6].set_xlabel("Measured Depth (m)")
    axes[6].set_ylabel("||Scale-1||")
    axes[6].grid(True)

    plt.suptitle(f"{term} Covariance Relative Scaling Validation")
    plt.tight_layout()
    plt.show()


def plot_reference_normalized_covariance_difference(
    df_out,
    calc_covs,
    excel_covs,
    term='DSTG',
    save_filename=None,
):
    """Plot covariance differences normalized by one benchmark reference scale.

    The common denominator is the largest absolute reference covariance element
    across all stations and the six unique NEV components for the selected term.
    Unlike pointwise percentage error, this remains finite when an individual
    reference component crosses or approaches zero.

    Args:
        df_out (pd.DataFrame): Trajectory dataframe containing ``MD``.
        calc_covs (dict): Calculated term covariance arrays.
        excel_covs (dict): Reference covariance dataframes.
        term (str): Error term to evaluate. Defaults to ``DSTG``.
        save_filename (str, optional): Output image path. If omitted, no file is saved.

    Returns:
        dict: Reference scale, largest absolute difference, normalized maximum,
              and the component and MD at which the maximum occurs.
    """
    components = ["NN", "EE", "VV", "NE", "NV", "EV"]
    if term not in calc_covs or term not in excel_covs:
        raise KeyError(f"Term '{term}' must exist in calculated and reference data.")

    calculated = np.asarray(calc_covs[term], dtype=float)
    reference_frame = excel_covs[term]
    expected_shape = (len(df_out), 3, 3)
    if calculated.shape != expected_shape:
        raise ValueError(
            f"Calculated term '{term}' has shape {calculated.shape}; "
            f"expected {expected_shape}."
        )
    if len(reference_frame) != len(df_out):
        raise ValueError(
            f"Reference term '{term}' has {len(reference_frame)} rows; "
            f"expected {len(df_out)}."
        )
    missing = set(components).difference(reference_frame.columns)
    if missing:
        raise ValueError(
            f"Reference term '{term}' is missing components: "
            + ", ".join(sorted(missing))
        )

    md = df_out["MD"].to_numpy(dtype=float)
    calculated_components = np.column_stack(
        [calculated[:,_COMPONENT_MAP[component][0],_COMPONENT_MAP[component][1],] for component in components]
    )
    reference_components = reference_frame[components].to_numpy(dtype=float)
    if not (
        np.isfinite(md).all()
        and np.isfinite(calculated_components).all()
        and np.isfinite(reference_components).all()
    ):
        raise ValueError(f"Term '{term}' contains non-finite plotting data.")

    reference_scale = float(np.max(np.abs(reference_components)))
    if reference_scale <= 0.0:
        raise ValueError(
            f"Term '{term}' has a zero reference scale and cannot be normalized."
        )

    differences = calculated_components - reference_components
    normalized_percent = 100.0 * differences / reference_scale
    maximum_index = np.unravel_index(
        np.argmax(np.abs(differences)), differences.shape
    )
    station_index, component_index = map(int, maximum_index)
    max_abs_difference = float(abs(differences[maximum_index]))
    max_normalized_abs = max_abs_difference / reference_scale

    fig, ax = plt.subplots(figsize=(11, 6))
    for component_index_plot, component in enumerate(components):
        ax.plot(
            md,
            normalized_percent[:, component_index_plot],
            linewidth=1.5,
            label=component,
        )
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Measured Depth (m)')
    ax.set_ylabel(
        r'$100\,\Delta C / \max_{k,c}|C^{\mathrm{ref}}_{k,c}|$ (\%)'
    )
    ax.set_title(f'{term} Reference-Scale-Normalized Covariance Difference')
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=3, loc='best')
    ax.text(
        0.02,
        0.98,
        (
            f"Reference scale: {reference_scale:.6e} m²\n"
            f"Maximum: {100.0 * max_normalized_abs:.6e}% "
            f"({components[component_index]} at MD {md[station_index]:.2f} m)"
        ),
        transform=ax.transAxes,
        va='top',
        ha='left',
        bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '0.7'},
    )
    fig.tight_layout()
    if save_filename:
        fig.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"-> Saved plot as {save_filename}")
    plt.show()

    summary = {
        'term': term,
        'reference_scale_m2': reference_scale,
        'max_abs_difference_m2': max_abs_difference,
        'max_normalized_abs': max_normalized_abs,
        'max_normalized_abs_percent': 100.0 * max_normalized_abs,
        'component_at_max': components[component_index],
        'md_at_max_m': float(md[station_index]),
    }
    print(f"\n[{term} Reference-Scale-Normalized Difference]")
    print(f"  Reference maximum : {reference_scale:.6e} m^2")
    print(f"  Maximum difference: {max_abs_difference:.6e} m^2")
    print(
        f"  Normalized maximum: {max_normalized_abs:.6e} "
        f"({100.0 * max_normalized_abs:.6e}%)"
    )
    print(
        f"  Location          : {components[component_index]} "
        f"at MD {md[station_index]:.2f} m"
    )
    return summary

def print_covariance_element_comparison(df_out, calc_covs, excel_covs, term, element, md_min, md_max):
    """
    Prints a terminal side-by-side comparison of a specific covariance element over a chosen MD range.

    Args:
            df_out (pd.DataFrame): Trajectory dataframe containing Measured Depths.
            calc_covs (dict): Dictionary of calculated covariance arrays.
            excel_covs (dict): Reference covariance DataFrames.
            term (str): Target ISCWSA term (e.g., 'DECR').
            element (str): Covariance matrix element (e.g., 'NV')
            md_min (float): Minimum measured depth interval to extract.
            md_max (float): Maximum measured depth interval to extract.
    """
    if element not in _COMPONENT_MAP:
        raise ValueError(f"Invalid element '{element}'. Choose from {list(_COMPONENT_MAP.keys())}")
        
    i, j = _COMPONENT_MAP[element]
    n = min(len(calc_covs[term]), len(excel_covs[term]))
    
    mds = df_out["MD"].to_numpy()[:n]
    mask = (mds >= md_min) & (mds <= md_max)
    filtered_mds = mds[mask]
    
    calc_values = calc_covs[term][:n, i, j][mask]
    excel_values = excel_covs[term][element].to_numpy()[:n][mask]
    
    print(f"\n{'-'*65}")
    print(f" {term} Covariance Element: {element} | MD Range: {md_min}m to {md_max}m ")
    print(f"{'-'*65}")
    print(f"{'MD (m)':>12} | {'Calculated':>15} | {'Excel Table':>15} | {'Difference':>15}")
    print(f"{'-'*65}")
    
    for md, calc, exc in zip(filtered_mds, calc_values, excel_values):
        diff = calc - exc
        print(f"{md:12.2f} | {calc:15.6e} | {exc:15.6e} | {diff:15.6e}")
    print(f"{'-'*65}\n")

