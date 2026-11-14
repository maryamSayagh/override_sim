
"""
HOVER — Evaluation Reporting

Purpose:
    Collect, format, and export evaluation results.

Supports:
    - Prediction metrics and probability recovery
    - Calibration results and reliability curves
    - Feature-group ablation
    - Within-world and cross-world generalization
    - Leakage validation records

Outputs:
    - CSV for tabular results
    - JSON for structured results
    - Markdown for readable summaries

No evaluation computations are performed here.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


# -------------------------------------------------------
# 1. Serialization helpers
# -------------------------------------------------------

def _serialize(value):
    """Convert NumPy and pandas values into JSON-safe types."""

    if isinstance(value, dict):
        return {
            str(k): _serialize(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return _serialize(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return _serialize(value.to_list())

    if isinstance(value, np.ndarray):
        return _serialize(value.tolist())

    if isinstance(value, np.generic):
        return _serialize(value.item())

    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()

    if value is None:
        return None

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def _output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# -------------------------------------------------------
# 2. Prediction metrics and probability recovery
# -------------------------------------------------------

def save_metrics_report(
    results,
    output_dir,
    name="metrics",
):
    """
    Save outputs from metrics.py.

    Accepts:
        dict returned by compute_metrics(),
        probability_recovery(), or evaluation_report().
    """

    path = _output_dir(output_dir)

    file = path / f"{name}.json"

    file.write_text(
        json.dumps(
            _serialize(results),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return file


# -------------------------------------------------------
# 3. Calibration reporting
# -------------------------------------------------------

def save_calibration_report(
    ece,
    reliability,
    output_dir,
    name="calibration",
):
    """
    Save ECE and reliability-curve coordinates.

    reliability:
        {
            "predicted": [...],
            "observed": [...]
        }
    """

    path = _output_dir(output_dir)

    data = {
        "ece": float(ece),
        "reliability_curve": reliability,
    }

    json_file = path / f"{name}.json"

    json_file.write_text(
        json.dumps(
            _serialize(data),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    curve_df = pd.DataFrame({
        "predicted": reliability["predicted"],
        "observed": reliability["observed"],
    })

    csv_file = path / f"{name}_curve.csv"

    curve_df.to_csv(csv_file, index=False)

    return {
        "json": json_file,
        "csv": csv_file,
    }


# -------------------------------------------------------
# 4. Ablation reporting
# -------------------------------------------------------

def save_ablation_report(
    results,
    output_dir,
    name="ablation",
):
    """Save full and summarized feature-group ablations."""

    path = _output_dir(output_dir)

    from .ablation import summarize_ablation

    summary = summarize_ablation(results)

    full_file = path / f"{name}_full.csv"
    summary_file = path / f"{name}_summary.csv"

    results.to_csv(full_file, index=False)
    summary.to_csv(summary_file, index=False)

    return {
        "full": full_file,
        "summary": summary_file,
    }


# -------------------------------------------------------
# 5. Generalization reporting
# -------------------------------------------------------

def save_generalization_report(
    results,
    output_dir,
    name="generalization",
):
    """
    Save within-world, cross-world (OOD),
    and leave-one-world-out results.
    """

    path = _output_dir(output_dir)

    full_file = path / f"{name}_full.csv"

    results.to_csv(full_file, index=False)

    protocol_files = {}

    for protocol, subset in results.groupby("protocol"):
        file = path / f"{name}_{protocol}.csv"
        subset.to_csv(file, index=False)
        protocol_files[protocol] = file

    return {
        "full": full_file,
        "protocols": protocol_files,
    }


# -------------------------------------------------------
# 6. Leakage validation reporting
# -------------------------------------------------------

def save_leakage_report(
    passed,
    output_dir,
    details=None,
    name="leakage",
):
    """
    Save leakage validation status.

    Call with passed=True only after validation succeeds.
    On validation exceptions, record passed=False and
    include the exception message in details.
    """

    path = _output_dir(output_dir)

    report = {
        "passed": bool(passed),
        "details": details or {},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    file = path / f"{name}.json"

    file.write_text(
        json.dumps(
            _serialize(report),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return file


# -------------------------------------------------------
# 7. Unified Markdown summary
# -------------------------------------------------------

def save_summary(
    sections,
    output_dir,
    name="evaluation_summary",
):
    """
    Create a readable Markdown summary.

    sections:
        {
            "Prediction Metrics": metrics_dict,
            "Calibration": calibration_dict,
            "Ablation": ablation_dataframe,
            "Generalization": generalization_dataframe,
        }
    """

    path = _output_dir(output_dir)

    lines = [
        "# HOVER Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    for title, data in sections.items():
        lines.extend([
            f"## {title}",
            "",
        ])

        if isinstance(data, pd.DataFrame):
            try:
                lines.append(data.to_markdown(index=False))
            except ImportError:
                lines.append(
                    "```text\n"
                    + data.to_string(index=False)
                    + "\n```"
                )

        elif isinstance(data, dict):
            lines.append("```json")
            lines.append(
                json.dumps(
                    _serialize(data),
                    indent=2,
                    allow_nan=False,
                )
            )
            lines.append("```")

        else:
            lines.append(str(data))

        lines.append("")

    file = path / f"{name}.md"

    file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return file
