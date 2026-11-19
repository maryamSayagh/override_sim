
"""
HOVER — Synthetic Dataset Generation Runner

Generates the three simulation worlds:
    W0, W_B, W_U

Exports for each world:
    data.csv       - Observable features + override target
    evaluation.csv - Ground truth and hidden simulator variables
    metadata.json  - Dataset provenance and configuration

Run from the parent directory of override_sim:

    python -m override_sim.hover.run_hover
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from override_sim.sim.generator import Simulator
from override_sim.config.schemas import SimConfig
from override_sim.config.worlds import (
    build_all_world_cfgs,
    get_world_description,
)
from override_sim.data.data_schema import (
    RAW_OBSERVABLE_COLS,
    TARGET_COL,
)


# -------------------------------------------------------
# Output paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "hover" / "simulation"


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def get_git_commit() -> str | None:
    """Return the current Git commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_safe(value):
    """Convert NumPy values to JSON-compatible values."""

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def validate_generated_dataset(df: pd.DataFrame, world: str):
    """Check structural consistency before exporting."""

    required = {
        "t",
        "user_id",
        "override",
        "y_true",
        "ai_rec",
        "y_final",
        "effective_override",
        "p_override_true",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{world}: Missing generated columns: {sorted(missing)}. "
            "Fix: Check sim/generator.py and the dataset assembly step."
        )

    if df.empty:
        raise ValueError(
            f"{world}: Empty dataset. "
            "Fix: Check n_events in the world configuration."
        )

    if df[["t", "user_id"]].isna().any().any():
        raise ValueError(
            f"{world}: Null decision identifiers. "
            "Fix: Check user assignment and event sequencing."
        )

    if df.duplicated(subset=["t", "user_id"]).any():
        raise ValueError(
            f"{world}: Duplicate (t, user_id) identifiers. "
            "Fix: Check event indexing in generator.py."
        )

    binary_cols = [
        "override",
        "y_true",
        "ai_rec",
        "y_final",
        "effective_override",
    ]

    for col in binary_cols:
        if not df[col].isin([0, 1]).all():
            raise ValueError(
                f"{world}: Invalid binary values in '{col}'. "
                "Fix: Inspect the corresponding decision logic "
                "in sim/generator.py."
            )

    p = df["p_override_true"]

    if p.isna().any() or not p.between(0, 1).all():
        raise ValueError(
            f"{world}: Invalid override probabilities. "
            "Fix: Check _override_probabilities() and "
            "override_intercept.py."
        )

    changed_action = (
        df["y_final"] != df["ai_rec"]
    ).astype(int)

    if not changed_action.equals(
        df["effective_override"].astype(int)
    ):
        raise ValueError(
            f"{world}: Inconsistent effective_override. "
            "Fix: Check final action calculation in generator.py."
        )

    if not changed_action.equals(df["override"].astype(int)):
        raise ValueError(
            f"{world}: Recorded override does not match "
            "the executed action. "
            "Fix: Check override eligibility and action execution "
            "in generator.py."
        )


def export_world(
    world_name: str,
    cfg: SimConfig,
    git_commit: str | None,
):
    """Generate and export one simulation world."""

    print(f"\nGenerating {world_name}...")

    simulator = Simulator(cfg)

    # Generate once so exported files refer to identical decisions.
    full_df = simulator.run(
        return_latents=True,
        hover_mode=False,
    )
    
    validate_generated_dataset(full_df, world_name)

    world_dir = OUTPUT_ROOT / world_name
    world_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------
    # Observable dataset
    # -----------------------------------------------

    observable_cols = [
        col
        for col in RAW_OBSERVABLE_COLS + [TARGET_COL]
        if col in full_df.columns
    ]

    data_df = full_df[observable_cols].copy()

    # -----------------------------------------------
    # Evaluation dataset
    # -----------------------------------------------

    evaluation_cols = [
        "t",
        "user_id",
        "y_true",
        "ai_rec",
        "override",
        "y_final",
        "effective_override",
        "p_override_true",
    ]

    # Include all available simulator internals for
    # diagnostic evaluation without exposing them to
    # prediction models.
    latent_cols = [
        col for col in full_df.columns
        if col.endswith("_latent")
        or col == "intercept_b"
    ]

    evaluation_cols = list(dict.fromkeys(
        evaluation_cols + latent_cols
    ))

    evaluation_df = full_df[evaluation_cols].copy()

    # -----------------------------------------------
    # Export CSV files
    # -----------------------------------------------

    data_path = world_dir / "data.csv"
    eval_path = world_dir / "evaluation.csv"

    data_df.to_csv(data_path, index=False)
    evaluation_df.to_csv(eval_path, index=False)

    # -----------------------------------------------
    # Reproducibility metadata
    # -----------------------------------------------

    metadata = {
        "world_id": world_name,
        "run_id": "001",
        "seed": cfg.seed,
        "description": get_world_description(world_name),
        "generator": {
            "module": "override_sim.sim.generator",
            "git_commit": git_commit,
            "config": json_safe(asdict(cfg)),
        },
        "dataset": {
            "file": "data.csv",
            "evaluation_file": "evaluation.csv",
            "n_rows": len(full_df),
            "n_users": int(full_df["user_id"].nunique()),
            "identifier_columns": ["t", "user_id"],
            "target_column": TARGET_COL,
        },
        "generation": {
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "target_override_rate": cfg.target_override_rate,
            "observed_override_rate": float(
                full_df["override"].mean()
            ),
            "disagreement_rate": float(
                full_df.attrs.get(
                    "disagreement_rate",
                    np.nan,
                )
            ),
            "intercept_b": float(
                full_df.attrs["cfg.intercept_b"]
            ),
        },
    }

    metadata_path = world_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(
            json_safe(metadata),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        f"  Rows: {len(full_df):,}\n"
        f"  Users: {full_df['user_id'].nunique():,}\n"
        f"  Target override rate: {cfg.target_override_rate:.3f}\n"
        f"  Observed override rate: {full_df['override'].mean():.3f}\n"
        f"  Saved: {world_dir}"
    )

    return {
        "world": world_name,
        "rows": len(full_df),
        "override_rate": float(full_df["override"].mean()),
        "output": str(world_dir),
    }


# -------------------------------------------------------
# Main runner
# -------------------------------------------------------

def main():

    base_cfg = SimConfig(
        target_override_rate=0.25
    )

    worlds = build_all_world_cfgs(base_cfg)
    git_commit = get_git_commit()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for world_name, cfg in worlds.items():
        result = export_world(
            world_name,
            cfg,
            git_commit,
        )
        results.append(result)

    print("\n=== Generation Summary ===")

    print(
        pd.DataFrame(results).to_string(
            index=False
        )
    )

    print("\nDatasets generated successfully.")


if __name__ == "__main__":
    main()
