import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from trajectory import calculate_spatial_trajectory 


# =====================================================================
# TRAJECTORY EVALUATION & VISUALIZATION
# =====================================================================
def calculate_spatial_error_metrics(df, residual_columns, md_col='MD'):
    """
    Calculate descriptive statistics for spatial-coordinate residuals.

    residual_columns:
        Mapping such as {
            'NS': 'NS_Deviation',
            'EW': 'EW_Deviation',
            'TVD': 'TVD_Deviation',
        }
    """
    md_offset = (
        df[md_col].to_numpy(dtype=float)
        - float(df[md_col].iloc[0])
    )

    records = []

    for coordinate, residual_col in residual_columns.items():
        residual = df[residual_col].to_numpy(dtype=float)

        if len(residual) > 1:
            slope_m_per_m = np.polyfit(
                md_offset,
                residual,
                deg=1,
            )[0]
        else:
            slope_m_per_m = np.nan

        records.append({
            'Coordinate': coordinate,
            'Mean_Signed_m': np.mean(residual),
            'MAE_m': np.mean(np.abs(residual)),
            'RMSE_m': np.sqrt(np.mean(residual**2)),
            'Max_Abs_m': np.max(np.abs(residual)),
            'Drift_mm_per_km': slope_m_per_m * 1e6,
            'EOW_Deviation_m': residual[-1],
        })

    return pd.DataFrame(records)

def evaluate_csv(file_path):
    """
    Reads a trajectory CSV, calculates coordinates, compares them to true values, 
    prints deviations, and plots the results.
    
    This ensures the minimum curvature implementation perfectly matches the 
    provided reference data.

    Args:
        file_path (str): The local file path to the target CSV survey file.

    Returns:
        None: This function handles stdout printing and matplotlib rendering directly.
    """
    # 1. Read the CSV data into a Pandas DataFrame
    df_true = pd.read_csv(file_path)
    
    # 2. Extract starting coordinates to ensure accurate tie-in calculation
    # This guarantees our computed trajectory originates from the exact same spatial location
    start_ns = df_true['NS'].iloc[0]
    start_ew = df_true['EW'].iloc[0]
    start_tvd = df_true['TVD'].iloc[0]
    
    # 3. Calculate the trajectory using the imported spatial function.
    # We rename the output columns to avoid overwriting the true CSV data for comparison.
    df_calc = calculate_spatial_trajectory(
        df_true, 
        ns_col='NS_calc', 
        ew_col='EW_calc', 
        tvd_col='TVD_calc',
        start_coords=(start_ns, start_ew, start_tvd)
    )
    
    # 4. Calculate deviations (Calculated - True) to find the absolute error at every station
    dev_ns = df_calc['NS_calc'] - df_calc['NS']
    dev_ew = df_calc['EW_calc'] - df_calc['EW']
    dev_tvd = df_calc['TVD_calc'] - df_calc['TVD']
    md = df_calc['MD']
    
    # 5. Extract EOW (End of Well) deviations (the final error accumulated at the bottom hole)
    eow_ns = dev_ns.iloc[-1]
    eow_ew = dev_ew.iloc[-1]
    eow_tvd = dev_tvd.iloc[-1]
    
    # 6. Find maximum deviations and the MD at which they occur
    # Using absolute values to find the largest magnitude of error regardless of direction
    max_ns_idx = dev_ns.abs().idxmax()
    max_ew_idx = dev_ew.abs().idxmax()
    max_tvd_idx = dev_tvd.abs().idxmax()

    df_calc['NS_Deviation'] = df_calc['NS_calc'] - df_calc['NS']
    df_calc['EW_Deviation'] = df_calc['EW_calc'] - df_calc['EW']
    df_calc['TVD_Deviation'] = df_calc['TVD_calc'] - df_calc['TVD']

    spatial_metrics = calculate_spatial_error_metrics(
        df_calc,
        {
            'NS': 'NS_Deviation',
            'EW': 'EW_Deviation',
            'TVD': 'TVD_Deviation',
        },
    )

    print(spatial_metrics.to_string(index=False))
    
    # 7. Print the analytical results to the console
    print("-" * 40)
    print("END OF WELL (EOW) DEVIATIONS")
    print("-" * 40)
    print(f"NS Deviation:  {eow_ns:.4f}")
    print(f"EW Deviation:  {eow_ew:.4f}")
    print(f"TVD Deviation: {eow_tvd:.4f}\n")
    
    print("-" * 40)
    print("MAXIMUM DEVIATIONS")
    print("-" * 40)
    print(f"Max NS Deviation:  {dev_ns.iloc[max_ns_idx]:.4f} at MD = {md.iloc[max_ns_idx]:.2f}")
    print(f"Max EW Deviation:  {dev_ew.iloc[max_ew_idx]:.4f} at MD = {md.iloc[max_ew_idx]:.2f}")
    print(f"Max TVD Deviation: {dev_tvd.iloc[max_tvd_idx]:.4f} at MD = {md.iloc[max_tvd_idx]:.2f}")
    
    # 8. Create the 3-panel plot to visually represent the error drift over the wellbore length
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # TVD Subplot
    ax1.plot(md, dev_tvd, color='blue')
    ax1.set_title('Trajectory Deviation Analysis (calculated minus supplied Volve coordinates)',fontweight="bold")
    ax1.set_ylabel('TVD Deviation (m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(['TVD Error (calculated - reference)'], loc='upper right')
    
    # NS Subplot
    ax2.plot(md, dev_ns, color='green', label='NS Error (Calc - reference)')
    ax2.set_ylabel('NS Deviation (m)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(['NS Error (calculated - reference)'], loc='upper right')
    
    # EW Subplot
    ax3.plot(md, dev_ew, color='red', label='EW Error (Calc - reference)')
    ax3.set_xlabel('Measured Depth (m)')
    ax3.set_ylabel('EW Deviation (m)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(['EW Error (calculated - reference)'], loc='upper right')
    
    # Adjust layout and display the finalized visualization
    plt.tight_layout()
    plt.show()

# =====================================================================
# SCRIPT EXECUTION
# =====================================================================
if __name__ == '__main__':
    # Define your CSV file path here
    csv_file = 'data/15_9_F_11_A.csv' 
    
    # Run the evaluation
    evaluate_csv(csv_file)