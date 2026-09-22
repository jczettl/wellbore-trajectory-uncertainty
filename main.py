import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# IMPORTS & DEPENDENCIES
# =====================================================================
from trajectory import ( 
    interpolate_trajectory_grid, 
    calculate_spatial_trajectory
)
from iscwsa_error import (
    calculate_iscwsa_covariance
)
from visualization import (
    plot_trajectory_2d, 
    plot_trajectory_3d_pyvista, 
    investigate_covariance_spikes, 
    plot_covariance_vs_md
)
from reporting import (
    generate_eow_report,
    generate_jacobian_comparison_table,
)
from survey_qc import (
    validate_and_clean_survey
)
from excel_extraction import (
    run_extraction
)
from dashboard import (
    plot_interactive_dashboard, 
    plot_static_dashboard, 
)
from covariance_analysis_visuals import(
    plot_ev_covariance_breakdown, 
    plot_sage_diagnostic,
    plot_sage_raw_vectors
)


# =====================================================================
# SELECTED ERROR-MODEL SUBSET
# =====================================================================
# This eleven-term configuration is used for the benchmark and demonstration
# cases. It is not a complete or tool-specific PUM.
SELECTED_ERROR_MODEL = {
    'DRFR':   {'mag': 0.35,        'axis': 'MD',  'mode': 'R', 'weight': lambda row, prev, dl: 1.0},
    'DSFS':   {'mag': 0.00056,     'axis': 'MD',  'mode': 'S', 'weight': lambda row, prev, dl: dl},
    'DSTG':   {'mag': 0.00000025,  'axis': 'MD',  'mode': 'G', 'weight': lambda row, prev, dl: (row['MD'] * row['TVD']) - (prev['MD'] * prev['TVD'])},
    'DEC-U':  {'mag': 0.16,        'axis': 'Azi', 'mode': 'W', 'weight': lambda row, prev, dl: 1.0},
    'DEC-OS': {'mag': 0.24,        'axis': 'Azi', 'mode': 'G', 'weight': lambda row, prev, dl: 1.0},
    'DEC-OH': {'mag': 0.21,        'axis': 'Azi', 'mode': 'G', 'weight': lambda row, prev, dl: 1.0},
    'DEC-OI': {'mag': 0.05,        'axis': 'Azi', 'mode': 'G', 'weight': lambda row, prev, dl: 1.0},
    'DECR':   {'mag': 0.1,         'axis': 'Azi', 'mode': 'R', 'weight': lambda row, prev, dl: 1.0},
    'SAGE':   {'mag': 0.2,         'axis': 'Inc', 'mode': 'S', 'weight': lambda row, prev, dl: max(0.0, np.sin(np.radians(row['Incl'])))**0.25},
    'XYM1':   {'mag': 0.1,         'axis': 'Inc', 'mode': 'S', 'weight': lambda row, prev, dl: np.abs(np.sin(np.radians(row['Incl'])))},
    'XYM2':   {'mag': 0.1,         'axis': 'Azi', 'mode': 'S', 'weight': lambda row, prev, dl: -1.0}
}


# =====================================================================
# ORCHESTRATION PIPELINE
# =====================================================================

def _save_matplotlib_plot(plot_function, output_path, *args, dpi=180, **kwargs):
    """Create, save, and close a Figure or (Figure, Axes) without showing it."""
    with plt.ioff():
        result = plot_function(*args, **kwargs)
        fig = result[0] if isinstance(result, tuple) else result
        # Optional diagnostics return None when their error term is absent.
        if fig is None:
            return
        try:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        finally:
            plt.close(fig)
    print(f"Saved plot: {output_path}")


def run_pipeline(
    data_path, 
    well_metadata, 
    start_coords=None, 
    dls_interval=30.0, 
    save_overview_plots=True, 
    generate_EOW_report=False, 
    section_borders_md=None,
    save_cov_dashboard=True, 
    save_ev_visuals=False,
    generate_jacobian_table=False,
    jacobian_table_output_dir='jacobian_comparison_results',
    balanced_tangential_jacobi_approximation=False,
    uncertainty_scale_mode='sigma',
    uncertainty_sigma_level=2.0,
    uncertainty_confidence=0.95,
    plots_output_dir='main_results',
    plot_dpi=180,
    save_pyvista=True,
    show_pyvista=False,
    show_interactive_dashboard=False,
):
    """
    Orchestrates the end-to-end execution of the wellbore trajectory calculation 
    and ISCWSA error propagation pipeline.
    
    It runs input QC, trajectory reconstruction, covariance propagation,
    reporting, and the requested visualizations.
    
    Args:
        data_path (str): File path to the target survey data (.csv or .xlsx).
        well_metadata (dict): Wellbore properties utilized for End-of-Well report headers.
        start_coords (tuple, optional): (North, East, TVD) physical tie-in point. Auto-extracted if None.
        dls_interval (float, optional): The standard interval length over which Dogleg Severity is normalized. Defaults to 30.0.
        save_overview_plots (bool, optional): Save standard trajectory and covariance plots. Defaults to True.
        generate_EOW_report (bool, optional): Toggles comprehensive text-based EOW report generation. Defaults to False.
        section_borders_md (list, optional): Measured depths marking casing shoes or kinematic section breaks. Defaults to None.
        save_cov_dashboard (bool, optional): Save the static covariance dashboard when section borders are supplied. Defaults to True.
        save_ev_visuals (bool, optional): Toggles diagnostic plots isolating EV component and SAGE dynamics. Defaults to False.
        generate_jacobian_table (bool): Generate the column-wise MCM-versus-BT
        jacobian_table_output_dir (str): Directory for the generated CSV and
            LaTeX table files.
        balanced_tangential_jacobi_approximation (bool): Use BT rather than full
            MCM displacement sensitivities.
        uncertainty_scale_mode (str): ``'sigma'`` (default) or ``'probability'``.
        uncertainty_sigma_level (float): Axis multiplier in sigma mode.
        uncertainty_confidence (float): Joint coverage in probability mode.
        plots_output_dir (str or Path): Output root, relative to the working
            directory unless absolute. Each input gets a subfolder named after
            its file stem. Existing plots with the same names are overwritten.
        plot_dpi (int): Resolution of saved Matplotlib PNGs. Defaults to 180.
        save_pyvista (bool): Save a 3D PNG when save_overview_plots is True.
            Defaults to True. Set False to use only the interactive viewer.
        show_pyvista (bool): Also open the interactive PyVista window when
            save_overview_plots is True. Defaults to False (save without a window).
        show_interactive_dashboard (bool): Open the interactive covariance
            dashboard, independently of the plot-saving flags. Defaults to
            False. This dashboard is displayed only and is never saved.
        
    Returns:
        None: This pipeline exclusively executes side-effects (file generation, plotting, console logging).
    """
    if not os.path.exists(data_path):
        print(f"[!] Skipping: File not found: {data_path}")
        return
        
    print(f"\n{'='*60}\n PROCESSING: {data_path}\n{'='*60}")

    # ---------------------------------------------------------------------
    # 1. DATA EXTRACTION & ERROR MODEL RESOLUTION
    # ---------------------------------------------------------------------
    if data_path.endswith('.xlsx'):
        print("Extracting Excel ISCWSA test data...")
        # Route Excel workbooks through the dedicated extraction wrapper to parse test benchmarks
        df_raw, active_error_model, _ = run_extraction(data_path, SELECTED_ERROR_MODEL)
        if df_raw is None:
            return
        # Benchmark files pass through the same finite-value and duplicate-MD
        # checks as field CSV input.
        df_qc = validate_and_clean_survey(df_raw)
        
    elif data_path.endswith('.csv'):
        print("Loading raw survey array...")
        df_raw = pd.read_csv(data_path)
        
        print("Running Quality Control Gate...")
        # Route standard CSV files through the Quality Control gate for validation
        df_qc = validate_and_clean_survey(df_raw)
        active_error_model = SELECTED_ERROR_MODEL
        
    else:
        print(f"Unsupported file format: {data_path}")
        return

    # Resolve and apply physical tie-in coordinates to accurately anchor the wellbore path
    if start_coords is None:
        tie_in_ns = df_qc['NS'].iloc[0] if 'NS' in df_qc.columns else 0.0
        tie_in_ew = df_qc['EW'].iloc[0] if 'EW' in df_qc.columns else 0.0
        tie_in_tvd = df_qc['TVD'].iloc[0] if 'TVD' in df_qc.columns else 0.0
        start_coords = (tie_in_ns, tie_in_ew, tie_in_tvd)
        print(f"Auto-extracted tie-in coordinates: NS={tie_in_ns}, EW={tie_in_ew}, TVD={tie_in_tvd}")

    # ---------------------------------------------------------------------
    # 2. KINEMATIC TRAJECTORY & SPATIAL ERROR PROPAGATION
    # ---------------------------------------------------------------------
    print("Building high-density grid using vector SLERP...")
    # Inject structurally interpolated stations to enhance visual and mathematical resolution
    df_dense = interpolate_trajectory_grid(df_qc, step_size=1)
    
    print("Processing structural path...")
    # Compute exact spatial displacement via Minimum Curvature stepping
    df_calculated = calculate_spatial_trajectory(
        df_dense, md_col='MD', inc_col='Incl', azi_col='Azi', 
        ns_col='NS', ew_col='EW', tvd_col='TVD', 
        start_coords=start_coords, dls_interval=dls_interval
    )
    
    print("Resolving ISCWSA error propagation...")
    # Dispatch the functional mathematical engine to accumulate spatial uncertainty covariances
    covariance_matrices, term_covariances, term_diagnostics = calculate_iscwsa_covariance(
        df_calculated, active_error_model, md_col='MD', inc_col='Incl', azi_col='Azi',balanced_tangential_jacobi_approximation=balanced_tangential_jacobi_approximation
    )

    # ---------------------------------------------------------------------
    # 3. OPERATIONAL END-OF-WELL (EOW) REPORTING
    # ---------------------------------------------------------------------
    if generate_EOW_report:
        print("\nGenerating End-Of-Well Report...")
        # Construct formatted text-based output integrating trajectory, uncertainty, and metadata
        generate_eow_report(
            df=df_calculated, 
            C_total_array=covariance_matrices, 
            dls_interval=dls_interval, 
            confidence=uncertainty_confidence,
            scale_mode=uncertainty_scale_mode,
            sigma_level=uncertainty_sigma_level,
            **well_metadata
        )

    input_stem = Path(data_path).stem
    plot_dir = Path(plots_output_dir) / input_stem

    if generate_jacobian_table:
        print("\nGenerating MCM-versus-BT Jacobian comparison table...")
        table_output_dir = plot_dir / jacobian_table_output_dir
        table_output_dir.mkdir(parents=True, exist_ok=True)
        generate_jacobian_comparison_table(
            df=df_calculated,
            output_dir=str(table_output_dir),
            file_stem=input_stem + "_mcm_vs_bt_jacobian_columns",
        )

    # ---------------------------------------------------------------------
    # 4. 2D/3D STRUCTURAL VISUALIZATION SUITE
    # ---------------------------------------------------------------------
    plot_dir = Path(plots_output_dir) / Path(data_path).stem
    if save_overview_plots or save_cov_dashboard or save_ev_visuals:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nPlot output directory: {plot_dir.resolve()}")

    if save_overview_plots:
        print("Saving visual diagnostic plots...")
        # Project error ellipses onto all three primary structural planes
        for index, plane in enumerate(['NE', 'NV', 'EV'], start=1):
            _save_matplotlib_plot(
                plot_trajectory_2d,
                plot_dir / f"{index:02d}_trajectory_{plane}.png",
                df_calculated,
                covariance_matrices,
                dpi=plot_dpi,
                plane=plane,
                ellipse_step=5,
                confidence=uncertainty_confidence,
                scale_mode=uncertainty_scale_mode,
                sigma_level=uncertainty_sigma_level,
            )
            
        # Plot general covariance matrix component trends
        _save_matplotlib_plot(
            plot_covariance_vs_md,
            plot_dir / "04_covariance_vs_md.png",
            df_calculated,
            covariance_matrices,
            dpi=plot_dpi,
        )
        _save_matplotlib_plot(
            investigate_covariance_spikes,
            plot_dir / "05_covariance_spikes.png",
            df_calculated,
            covariance_matrices,
            section_borders_md=section_borders_md,
            dpi=plot_dpi,
        )

    # ---------------------------------------------------------------------
    # 5. DIAGNOSTIC DASHBOARDS & VECTOR ANALYTICS
    # ---------------------------------------------------------------------
    if save_cov_dashboard and section_borders_md is not None:
        print("Saving static covariance dashboard...")
        _save_matplotlib_plot(
            plot_static_dashboard,
            plot_dir / "07_covariance_dashboard_sections.png",
            df_calculated,
            covariance_matrices,
            section_borders_md=section_borders_md,
            dpi=plot_dpi,
        )

    if save_ev_visuals:
        print("Saving EV and SAGE diagnostics...")
        # Deconstruct EV coupling and map SAGE kinematic cancellation events
        _save_matplotlib_plot(
            plot_ev_covariance_breakdown,
            plot_dir / "08_ev_covariance_breakdown.png",
            df=df_calculated, 
            C_total_array=covariance_matrices, 
            term_covariances=term_covariances, 
            error_model=active_error_model,
            dpi=plot_dpi,
        )
        _save_matplotlib_plot(
            plot_sage_diagnostic,
            plot_dir / "09_sage_diagnostic.png",
            df=df_calculated,
            term_covariances=term_covariances,
            dpi=plot_dpi,
        )
        _save_matplotlib_plot(
            plot_sage_raw_vectors,
            plot_dir / "10_sage_raw_vectors.png",
            df_calculated,
            term_diagnostics,
            dpi=plot_dpi,
        )

    # Open interactive views after saving the static figures.
    if show_interactive_dashboard:
        print("Opening interactive covariance dashboard...")
        fig_dashboard = plot_interactive_dashboard(df_calculated, covariance_matrices)
        try:
            plt.show(block=True)
        finally:
            plt.close(fig_dashboard)

    if save_overview_plots and (save_pyvista or show_pyvista):
        print("Preparing 3D cross-hole uncertainty view...")
        try:
            plot_trajectory_3d_pyvista(
                df_calculated,
                covariance_matrices,
                confidence=uncertainty_confidence,
                scale_mode=uncertainty_scale_mode,
                sigma_level=uncertainty_sigma_level,
                output_path=plot_dir / "11_trajectory_3d.png" if save_pyvista else None,
                show=show_pyvista,
            )
        except ImportError as exc:
            print(f"[!] Skipping PyVista plot: {exc}")

    print(f"\n Finished processing: {data_path}")


# =====================================================================
# BATCH EXECUTION MODULE
# =====================================================================
if __name__ == "__main__":

    # The physical files targeted for sequential pipeline execution
    TARGET_FILES = [
        'data/15_9_F_11_A.csv'
        #'data/error-model-example-mwdrev5-1-iscwsa-1.xlsx',
        #'data/error-model-example-mwdrev5-1-iscwsa-2.xlsx',
        #'data/error-model-example-mwdrev5-1-iscwsa-3.xlsx'
    ]

    # Metadata dictionary mapping survey properties to their respective files
    metadata_map = {
        '15_9_F_11_A.csv': {
            "operator": "Equinor (formerly Statoil)",
            "well": "15/9-F-11 A",
            "field": "Volve Field (Central North Sea)",
            "site": "Maersk Inspirer",
            "country_state": "Norway / Offshore",
            "api_uwi": "NPDID Wellbore ID: 7079",
            "geodetic_datum": "ED50 (European Datum 1950)",
            "north_ref": "Grid North",
            "system_grid": "ED50 / UTM Zone 31N (EPSG 23031)",
            "convergence": -0.95,
            "declination": -2.10,
            "elevation_ref": "RKB (Rotary Kelly Bushing)",
            "measured_from": "RKB (approx. 40-70m above MSL)",
            "survey_method": "Minimum Curvature",
            "error_model": "Mini ISCWSA",
            "section_borders_md": [620.0, 2130.0, 3168.1, 3571.7]
        }
    }

    # Iteratively run the orchestration pipeline for each queued target file
    for target in TARGET_FILES:

        well_meta = metadata_map.get(Path(target).name, {})
        
        section_borders_md = well_meta.get("section_borders_md", None)

        run_pipeline(
            data_path=target, 
            well_metadata=well_meta, 
            section_borders_md=section_borders_md,
            start_coords=None, 
            dls_interval = 30.0,

            balanced_tangential_jacobi_approximation=True, #if false uses MCM jacobian for uncertaintys
            uncertainty_scale_mode='sigma', 
            uncertainty_sigma_level=2.0, #if mode 'probability' use uncertainty_confidence

            save_overview_plots=True,
            save_cov_dashboard=True, 
            save_pyvista=True, 
            save_ev_visuals=False,

            generate_EOW_report=True,
            generate_jacobian_table=True,

            show_pyvista=True, 
            show_interactive_dashboard=False, 

            plots_output_dir='main_results',
            plot_dpi=180,
        )
