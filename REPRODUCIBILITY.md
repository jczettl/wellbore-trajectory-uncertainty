# Reproducing the Report

This guide maps every table and figure in [Report.pdf](Report.pdf) to its input data, calculation, script, and output. Run all commands from the repository root after installing `requirements.txt`.

## Data and fixed settings

| Input | Source and contents | Used for |
| --- | --- | --- |
| [`15_9_F_11_A.csv`](15_9_F_11_A.csv) | Survey from the [Equinor Volve dataset](https://www.equinor.com/energy/volve-data-sharing), supplied as MD, inclination, azimuth, and reference TVD/North/East coordinates. It contains 323 physical stations from 145.9 to 3762.0 m MD. | Coordinate verification, Jacobian checks, MCM/BT comparison, and the uncertainty case study. |
| `error-model-example-mwdrev5-1-iscwsa-1.xlsx`, `...-2.xlsx`, `...-3.xlsx` | Three [ISCWSA Revision 5 benchmark workbooks](https://www.iscwsa.net/error-model-documentation/). `excel_extraction.py` reads their `Wellpath`, `Model`, and selected error-term diagnostic sheets. | Spreadsheet trajectory and covariance verification. |

Keep the benchmark filenames unchanged. `excel_extraction.py` recognizes the feet-based second example from `iscwsa-2` in the filename and converts its depths to metres.

The report uses these settings:

- MCM trajectory reconstruction in every configuration.
- BT sensitivities for the main Volve case; analytical MCM sensitivities are evaluated as a comparison.
- The eleven terms in `main.SELECTED_ERROR_MODEL`. The equivalent configuration in `jacobian_validation.STANDARD_ERROR_MODEL` is used by `mcm_bt_comparison.py`.
- Covariance stored at the 1σ level and uncertainty axes displayed with `scale_mode='sigma'`, `sigma_level=2.0`.
- A 1 m SLERP grid in `main.py` for visualization; covariance accumulation at physical survey anchors.
- DLS normalized to 30 m.
- Finite-difference steps of `1e-3` degrees for inclination and azimuth and `0.1` m for course length. Convergence multipliers are `(10, 1, 0.1, 0.01, 0.001)`.
- MCM/BT NE-orientation comparison only when both ellipses have relative anisotropy of at least `0.01`; angles are wrapped modulo 180 degrees.
- Volve case-study section boundaries at MD 620.0, 2130.0, 3168.1, and 3571.7 m.

A 2σ axis multiplier is a display convention and is not a dimension-independent 95% joint probability region.

## Commands

| Purpose | Command | Result |
| --- | --- | --- |
| Volve case study | `python main.py` | End-of-well console summary and plots under `main_results/15_9_F_11_A/`. |
| Volve coordinate comparison | `python csv_verification.py` | Console statistics and an interactive residual plot. |
| Per-workbook ISCWSA checks | `python excel_verification.py` | TVD and covariance statistics for each example. |
| Aggregated benchmark tables | `python benchmark_summary.py` | Report Tables 3 and 4 in the console. |
| ISCWSA TVD plots | Set `plot_tvd=True` in the call at the bottom of `excel_verification.py`, then run `python excel_verification.py`. | One displayed TVD-residual figure per workbook. |
| Analytical-versus-FD checks | `python jacobian_validation.py` | Console errors, worst intervals, and step-size convergence. |
| FD verification plots | `python jacobian_validation_plot.py` | Two PNGs in `jacobian_validation_plot/`. |
| MCM-versus-BT tables | `python mcm_bt_comparison.py` | Report Tables 6, 7, and 10 in the console. |
| MCM-versus-BT plot | `python mcm_bt_comparison_plot.py` | One PNG in `mcm_bt_comparison_plot/`. |

Close interactive Matplotlib windows to let their scripts continue. The Excel verification reports numerical residuals without applying an arbitrary pass/fail tolerance.

## Report tables

Tables 1, 2, and 9 contain manually assembled configuration or descriptive information. The remaining tables summarize calculated data. Report typesetting and rounding are separate from console formatting.

| Table | Data and calculation source | Reproduction |
| --- | --- | --- |
| **1 - Selected error terms** | Codes and magnitudes from the workbook `Model` sheets; axes, correlation modes, and weights from the error-model dictionaries. | Inspect `main.SELECTED_ERROR_MODEL`, `jacobian_validation.STANDARD_ERROR_MODEL`, and `excel_extraction.py`. DSFS and DSTG are printed as cumulative weights in the report; the implementation applies segment differences. |
| **2 - Error-term descriptions** | Explanatory descriptions of the eleven selected ISCWSA terms based on the references cited in the report. | Read Table 2 and the report references. This table is authored text rather than a program output. |
| **3 - ISCWSA TVD residuals** | MCM-reconstructed TVD minus each workbook's reference `Wellpath` TVD. | Run `python benchmark_summary.py`. The table contains maximum absolute residual, its MD, signed EOW residual, and total MD. |
| **4 - Covariance benchmark** | Every selected term's calculated covariance compared with its workbook diagnostic sheet, using both sensitivity formulations. | Run `python benchmark_summary.py`. For each term and method it prints the largest absolute difference across all three wells, all stations, and the six independent covariance elements. |
| **5 - Volve coordinate residuals** | `csv_verification.calculate_spatial_error_metrics`, using supplied `NS`, `EW`, and `TVD` as references. | Run `python csv_verification.py`; use mean signed, MAE, RMSE, maximum absolute, and signed EOW residuals. |
| **6 - Jacobian columns** | `reporting.generate_jacobian_comparison_table`, called by `mcm_bt_comparison.py`, over 322 physical legs. | Run `python mcm_bt_comparison.py`. Each sample is the Euclidean norm of the MCM-minus-BT NEV sensitivity vector for one parameter column. |
| **7 - Derived uncertainty** | `mcm_bt_comparison.compare_uncertainty`, using MCM and BT covariance profiles for the same Volve trajectory and error model. | Run `python mcm_bt_comparison.py` with the default 2σ scaling and `0.01` anisotropy threshold. |
| **8 - End-of-well summary** | Final reconstructed coordinates and covariance passed by `main.py` to `reporting.generate_eow_report`. | Run `python main.py`. Horizontal axes and angle come from the NE covariance; vertical extent is `2 * sqrt(C_VV)`. |
| **9 - Descriptive sections** | Manually selected boundaries from `main.py`'s `metadata_map`, plus final MD. End TVDs come from the supplied Volve coordinates at those MDs. | Look up MD 620.0, 2130.0, 3168.1, 3571.7, and 3762.0 m in the CSV and round reference TVD to one decimal. Section names and descriptions are authored interpretations. |
| **10 - Covariance elements** | `mcm_bt_comparison.compare_covariance_elements`, using the six independent elements of both covariance arrays. | Run `python mcm_bt_comparison.py`; it prints RMS, maximum, global BT-referenced relative RMS, signed EOW difference, and MD at maximum. |

## Report figures

For `main.py`, the default output directory is `main_results/15_9_F_11_A/`. Multi-panel layouts and selected 3D viewpoints were assembled for the report; the scripts generate the component figures rather than rebuilding the PDF.

| Figure | Script and function | Output or setting |
| --- | --- | --- |
| **1 - ISCWSA TVD residuals** | `excel_verification.py` calls `excel_verification_visuals.plot_tvd_validation` for all three workbooks. | Set `plot_tvd=True` in the final `run_verification` call and run `python excel_verification.py`; save the three displayed plots. Uses calculated-minus-reference TVD without baseline removal. |
| **2 - Volve coordinate residuals** | `csv_verification.evaluate_csv`. | Run `python csv_verification.py`; save the displayed three-panel figure. The README copy is `figures/volve_coordinate_residuals.png`. |
| **3a/3b - FD verification and convergence** | `jacobian_validation_plot.py` uses `jacobian_validation.run_validation`. | `jacobian_validation_plot/01_fd_jacobian_error_vs_md.png` and `jacobian_validation_plot/06_fd_convergence.png`. Default representative legs are 1, 107, 214, and 322. |
| **4 - Propagated MCM/BT differences** | `mcm_bt_comparison_plot.plot_comparison` uses `mcm_bt_comparison.compare_uncertainty`. | `mcm_bt_comparison_plot/propagated_uncertainty_difference.png`; 2σ axes and anisotropy threshold 0.01. |
| **5 - Sectioned trajectory/covariance dashboard** | `main.py` calls `dashboard.plot_static_dashboard`. | `07_covariance_dashboard_sections.png`; requires `save_cov_dashboard=True` and the boundaries in `metadata_map`. |
| **6a/6b/6c - NE, NV, and EV projections** | `main.py` calls `visualization.plot_trajectory_2d`. | `01_trajectory_NE.png`, `02_trajectory_NV.png`, and `03_trajectory_EV.png`; requires `save_overview_plots=True`. Ellipses are shown at every fifth physical station. |
| **7 - 3D trajectory and EOW close-up** | `main.py` calls `visualization.plot_trajectory_3d_pyvista`. | `11_trajectory_3d.png` with `save_pyvista=True`. The report's additional EOW view requires selecting another camera position; set `show_pyvista=True` to inspect it interactively. |
| **8 - Variance-growth diagnostic** | `main.py` calls `visualization.investigate_covariance_spikes`. | `05_covariance_spikes.png`; requires `save_overview_plots=True`. |
| **9 - Full covariance diagnostic** | `main.py` calls `visualization.plot_covariance_vs_md`. | `04_covariance_vs_md.png`; requires `save_overview_plots=True`. The README copy is `figures/covariance_vs_md.png`. |
| **10 - EV covariance by error term** | `main.py` calls `covariance_analysis_visuals.plot_ev_covariance_breakdown`. | `08_ev_covariance_breakdown.png`; set `save_ev_visuals=True`. This also produces two SAGE diagnostic plots. |

The README's NE preview is a copy of `01_trajectory_NE.png` saved as `figures/trajectory_NE.png`.

## Direct project-module dependencies

| File | Direct project imports |
| --- | --- |
| `main.py` | `covariance_analysis_visuals.py`, `dashboard.py`, `excel_extraction.py`, `iscwsa_error.py`, `reporting.py`, `survey_qc.py`, `trajectory.py`, `visualization.py` |
| `trajectory.py` | `core_math.py` |
| `core_math.py` | None |
| `iscwsa_error.py` | `core_math.py` |
| `survey_qc.py` | None |
| `reporting.py` | `core_math.py` |
| `visualization.py` | `core_math.py` |
| `dashboard.py` | None |
| `covariance_analysis_visuals.py` | None |
| `excel_extraction.py` | None |
| `csv_verification.py` | `trajectory.py` |
| `excel_verification.py` | `excel_extraction.py`, `excel_verification_metrics.py`, `excel_verification_visuals.py`, `iscwsa_error.py`, `trajectory.py` |
| `benchmark_summary.py` | `excel_verification.py`, `excel_verification_metrics.py` |
| `excel_verification_metrics.py` | None |
| `excel_verification_visuals.py` | None |
| `jacobian_validation.py` | `core_math.py`, `survey_qc.py` |
| `jacobian_validation_plot.py` | `jacobian_validation.py` |
| `mcm_bt_comparison.py` | `core_math.py`, `iscwsa_error.py`, `jacobian_validation.py`, `reporting.py`, `survey_qc.py`, `trajectory.py` |
| `mcm_bt_comparison_plot.py` | `mcm_bt_comparison.py` |

## Known report/code rounding differences

- The current code prints an EOW NE semi-major axis of **11.00 m** after rounding to two decimals; report Table 8 states **11.01 m**. This value should be reconciled before releasing a revised report.
- Table 9 uses reference TVD values from the CSV. At MD 3571.7 m, the reference is 2980.65 m while the reconstructed value is approximately 2980.6454 m, which can change one-decimal rounding.
- Very small finite-difference and near-zero covariance residuals can vary with the numerical environment.

The README previews and `benchmark_summary.py` were checked with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, Matplotlib 3.10.8, and openpyxl 3.1.5. These are reproduction versions, not a record of the original report's software environment.
