"""Print the aggregated ISCWSA benchmark results used in report Tables 3 and 4.

Run this file from the repository root:
    python benchmark_summary.py

The three benchmark workbooks must retain their repository filenames because
the extractor uses the ``iscwsa-2`` name to identify the feet-based example.
"""

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd

from excel_verification import run_pipeline
from excel_verification_metrics import covariance_max_absolute_error


BENCHMARK_FILES = tuple(
    f"data/error-model-example-mwdrev5-1-iscwsa-{number}.xlsx"
    for number in (1, 2, 3)
)

REPORT_TERM_ORDER = (
    "DEC-OS", "DEC-OH", "DEC-U", "SAGE", "XYM2", "XYM1",
    "DSTG", "DEC-OI", "DSFS", "DRFR", "DECR",
)


def run_benchmark(file_path, use_bt):
    """Run one existing benchmark pipeline while keeping this summary concise."""
    log = StringIO()
    try:
        with redirect_stdout(log):
            result = run_pipeline(
                file_path,
                balanced_tangential_jacobi_approximation=use_bt,
            )
    except Exception:
        print(log.getvalue(), end="")
        raise
    if result is None:
        raise RuntimeError(f"Benchmark extraction failed: {file_path}")
    return result


def collect_report_tables(file_paths=BENCHMARK_FILES):
    """Return report Table 3 and Table 4 as pandas DataFrames."""
    tvd_rows = []
    covariance_rows = []

    for method, use_bt in (("MCM", False), ("BT", True)):
        for example, file_path in enumerate(file_paths, start=1):
            print(f"Processing ISCWSA-{example} with {method} sensitivities...")
            reference, calculated, terms, reference_terms = run_benchmark(file_path, use_bt)

            errors = covariance_max_absolute_error(terms, reference_terms)
            covariance_rows.append(
                errors[["Error Term", "Overall Max Diff"]].assign(Method=method)
            )

            # Trajectory integration is MCM in both sensitivity configurations.
            if method == "MCM":
                residual = calculated["TVD"].to_numpy() - reference["TVD"].to_numpy()
                maximum = int(np.argmax(np.abs(residual)))
                tvd_rows.append({
                    "Example": f"ISCWSA-{example}",
                    "Max abs (m)": abs(residual[maximum]),
                    "MD at max (m)": calculated["MD"].iloc[maximum],
                    "EOW (m)": residual[-1],
                    "Total MD (m)": calculated["MD"].iloc[-1],
                })

    table_3 = pd.DataFrame(tvd_rows)
    table_4 = (
        pd.concat(covariance_rows)
        .groupby(["Error Term", "Method"])["Overall Max Diff"]
        .max()
        .unstack("Method")[["MCM", "BT"]]
        .reindex(REPORT_TERM_ORDER)
    )
    return table_3, table_4


def main():
    table_3, table_4 = collect_report_tables()
    print("\nTABLE 3 - ISCWSA TVD RESIDUALS")
    print(table_3.to_string(index=False, float_format="{:.6e}".format))
    print("\nTABLE 4 - MAXIMUM COVARIANCE-ELEMENT DIFFERENCE (m^2)")
    print(table_4.to_string(float_format="{:.6e}".format))


if __name__ == "__main__":
    main()
